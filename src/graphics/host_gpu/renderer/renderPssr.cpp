#include "common/assert.h"
#include "common/threads.h"
#include "gpu_blit_shaders/gpu_pssr_linear_spv.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <array>
#include <mutex>
#include <optional>

namespace Libs::Graphics {
namespace {

struct PssrParameters {
	uint32_t input_size[2];
	uint32_t output_size[2];
	uint32_t output_channels[4];
};

// DLSS reads and writes with stages and accesses of its own.
void MemoryBarrier(vk::CommandBuffer command) {
	const vk::MemoryBarrier2 barrier {.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
	                                  .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
	                                  .dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
	                                  .dstAccessMask = vk::AccessFlagBits2::eMemoryRead |
	                                                   vk::AccessFlagBits2::eMemoryWrite};
	command.pipelineBarrier2(
	    vk::DependencyInfo {.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}

// The execution thread chooses quality modes and releases features, and the thread that records
// the Vulkan command buffer creates and evaluates them: NGX is called by one of them at a time.
std::mutex g_ngx_mutex;

// The linear upscale, recorded on its own or where DLSS cannot take the request.
struct LinearPssr {
	vk::PipelineLayout layout;
	vk::Pipeline       pipeline;
	vk::ImageView      input;
	vk::ImageView      output;
	PssrParameters     parameters;

	void Record(vk::CommandBuffer command) const {
		const std::array images {
		    vk::DescriptorImageInfo {nullptr, input, vk::ImageLayout::eGeneral},
		    vk::DescriptorImageInfo {nullptr, output, vk::ImageLayout::eGeneral}};
		std::array<vk::WriteDescriptorSet, 2> writes {};
		for (uint32_t i = 0; i < writes.size(); ++i) {
			writes[i].dstBinding = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType = i == 0 ? vk::DescriptorType::eSampledImage
			                                 : vk::DescriptorType::eStorageImage;
			writes[i].pImageInfo = &images[i];
		}
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes.size(),
		                             writes.data());
		command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
		command.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(parameters),
		                      &parameters);
		command.dispatch((parameters.output_size[0] + 7) / 8, (parameters.output_size[1] + 7) / 8,
		                 1);
		ShaderAccessBarrier(CommandRecorder {command}, vk::PipelineStageFlagBits::eComputeShader);
	}
};

} // namespace

// The DLSS feature of a PSSR context. The execution thread makes one when a request needs a new
// feature; NGX creates and evaluates it where the PSSR command is recorded into a Vulkan command
// buffer: at once with a direct recorder, on the recording thread with a deferred one. Its fields
// are accessed under g_ngx_mutex.
struct RenderExecutor::DlssFeature {
	Dlss::Creation creation;
	Dlss::Feature* feature = nullptr; // null until created, and when NGX refused it
	bool           created = false;
	bool           failed  = false; // an evaluation failed
};

RenderExecutor::~RenderExecutor() {
	for (const auto& context: m_dlss_contexts) {
		ReleaseDlssFeature(context.feature);
	}
	const auto device = m_context.GetGraphics().device;
	device.destroyPipeline(m_pssr_pipeline, nullptr);
	device.destroyPipelineLayout(m_pssr_pipeline_layout, nullptr);
	device.destroyDescriptorSetLayout(m_pssr_descriptor_layout, nullptr);
}

RenderExecutor::DlssFeature* RenderExecutor::DlssFeatureFor(const Pssr::Request& request,
                                                            const Dlss::Inputs&  inputs,
                                                            bool&                reset) {
	// After the GPU is done with the commands recorded until now.
	const auto release = [&](DlssFeature* feature) {
		m_context.GetCommandScheduler().DeferOperation([feature] { ReleaseDlssFeature(feature); });
	};
	auto context = std::ranges::find(m_dlss_contexts, request.context, &DlssContext::context);
	reset        = (request.flags & Pssr::kReset) != 0;
	if (context == m_dlss_contexts.end() || !Dlss::Takes(context->creation, request, inputs)) {
		// A render size no quality mode takes scales linearly.
		std::optional<Dlss::Creation> creation;
		{
			std::lock_guard lock(g_ngx_mutex);
			creation = Dlss::Choose(*Dlss::Capabilities(), request, inputs);
		}
		if (!creation) {
			return nullptr;
		}
		if (context != m_dlss_contexts.end()) {
			release(context->feature);
		} else {
			// A library has at most four contexts.
			if (m_dlss_contexts.size() == 4) {
				release(m_dlss_contexts.front().feature);
				m_dlss_contexts.erase(m_dlss_contexts.begin());
			}
			context = m_dlss_contexts.insert(m_dlss_contexts.end(), {request.context, {}, nullptr});
		}
		context->creation = *creation;
		context->feature  = new DlssFeature {.creation = *creation};
		reset             = true;
	}
	return context->feature;
}

bool RenderExecutor::EvaluateDlss(vk::CommandBuffer command, DlssFeature& feature,
                                  const Pssr::Request& request, Dlss::Inputs& inputs, bool reset) {
	std::lock_guard lock(g_ngx_mutex);
	if (!feature.created) {
		feature.created = true;
		feature.feature = Dlss::CreateFeature(command, feature.creation);
	}
	if (feature.feature == nullptr || feature.failed) {
		return false;
	}
	MemoryBarrier(command);
	if (!Dlss::Evaluate(command, feature.feature, request, inputs, reset)) {
		feature.failed = true;
		return false;
	}
	MemoryBarrier(command);
	return true;
}

void RenderExecutor::ReleaseDlssFeature(DlssFeature* feature) {
	if (feature == nullptr) {
		return;
	}
	{
		std::lock_guard lock(g_ngx_mutex);
		if (feature->feature != nullptr) {
			Dlss::ReleaseFeature(feature->feature);
		}
	}
	delete feature;
}

void RenderExecutor::CreatePssrPipeline() {
	auto& graphics = m_context.GetGraphics();
	const std::array bindings {
	    vk::DescriptorSetLayoutBinding {0, vk::DescriptorType::eSampledImage, 1,
	                                    vk::ShaderStageFlagBits::eCompute},
	    vk::DescriptorSetLayoutBinding {1, vk::DescriptorType::eStorageImage, 1,
	                                    vk::ShaderStageFlagBits::eCompute}};
	vk::DescriptorSetLayoutCreateInfo descriptor_info {};
	descriptor_info.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptor_info.bindingCount = bindings.size();
	descriptor_info.pBindings = bindings.data();
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(
	                         &descriptor_info, nullptr, &m_pssr_descriptor_layout),
	                     "create upscaling descriptor layout");
	const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0,
	                                  sizeof(PssrParameters)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts = &m_pssr_descriptor_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges = &push;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(
	                         &layout_info, nullptr, &m_pssr_pipeline_layout),
	                     "create upscaling pipeline layout");
	const auto shader = CompileSPV(GPU_PSSR_LINEAR_SPV, graphics.device);
	vk::ComputePipelineCreateInfo create {};
	create.stage.stage = vk::ShaderStageFlagBits::eCompute;
	create.stage.module = shader;
	create.stage.pName = "main";
	create.layout = m_pssr_pipeline_layout;
	const auto result = graphics.device.createComputePipelines(
	    nullptr, 1, &create, nullptr, &m_pssr_pipeline);
	graphics.device.destroyShaderModule(shader, nullptr);
	RequireVulkanSuccess(result, "create upscaling pipeline");
}

void RenderExecutor::ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request) {
	m_context.GetCommandScheduler().PopPendingOperations();
	Common::LockGuard lock(m_context.GetMutex());
	ShaderTextureResource input_descriptor {}, output_descriptor {};
	std::copy_n(request.input, 8, input_descriptor.fields);
	std::copy_n(request.output, 8, output_descriptor.fields);
	const auto is_2d = [](const ShaderTextureResource& descriptor) {
		return !descriptor.IsNull() &&
		       (descriptor.Type() == Prospero::ImageType::kColor2D ||
		        descriptor.Type() == Prospero::ImageType::kColor2DArray) &&
		       descriptor.TileMode() != Prospero::TileMode::kDepth;
	};
	EXIT_IF(!is_2d(input_descriptor) || !is_2d(output_descriptor) ||
	        !Pssr::IsColorFormat(input_descriptor.Format()) ||
	        output_descriptor.Format() != Prospero::BufferFormat::k11_11_10Float ||
	        output_descriptor.TileMode() != Prospero::TileMode::kStandard256B ||
	        output_descriptor.BaseLevel() != 0 || output_descriptor.LastLevel() != 0 ||
	        output_descriptor.MaxMip() != 0 || output_descriptor.BaseArray5() != 0 ||
	        output_descriptor.Depth() != 0 || request.width == 0 || request.height == 0);
	const auto input_width = std::max(1u, (input_descriptor.Width5() + 1u) >>
	                                         input_descriptor.BaseLevel());
	const auto input_height = std::max(1u, (input_descriptor.Height5() + 1u) >>
	                                          input_descriptor.BaseLevel());
	EXIT_IF(request.width > input_width || request.height > input_height);
	PssrParameters parameters {{request.width, request.height},
	                           {output_descriptor.Width5() + 1u, output_descriptor.Height5() + 1u},
	                           {4, 4, 4, 4}};
	for (uint32_t component = 0; component < 3; ++component) {
		for (uint32_t source = 0; source < 4; ++source) {
			if (GetDstSel(output_descriptor.DstSelXYZW(), source) == 4u + component) {
				parameters.output_channels[component] = source;
				break;
			}
		}
	}
	const auto resolve = [&](const uint32_t* words, bool output) {
		ShaderRecompiler::IR::ImageResource resource {};
		resource.resource_class = output ? ShaderRecompiler::IR::ImageResourceClass::Storage
		                                 : ShaderRecompiler::IR::ImageResourceClass::Sampled;
		resource.numeric_class = Prospero::TextureNumericClass::Float;
		resource.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
		resource.read = !output;
		resource.written = output;
		ShaderRecompiler::IR::DescriptorValue value {};
		std::copy_n(words, 8, value.dwords.begin());
		value.dword_count = 8;
		auto binding = ResolveTexture(resource, value);
		binding.desc.view_info.level_count = 1;
		binding.desc.view_info.min_lod = 0;
		return binding;
	};
	auto& cache = m_context.GetTextureCache();
	auto input = resolve(request.input, false);
	BindImage(input.image_id, false);
	auto output = resolve(request.output, true);
	BindImage(output.image_id, true);
	EXIT_IF(input.desc.info.data.address < output.desc.info.data.End() &&
	        output.desc.info.data.address < input.desc.info.data.End());
	// Depth, motion vectors and exposure for DLSS, which writes the output unswizzled.
	const std::array<const uint32_t*, 3>         dlss_words {request.depth, request.motion,
	                                                         request.exposure};
	std::array<std::optional<TextureBinding>, 3> dlss_bindings;
	if (Dlss::Capabilities() != nullptr && Dlss::Accepts(request) &&
	    parameters.output_channels[0] == 0 && parameters.output_channels[1] == 1 &&
	    parameters.output_channels[2] == 2) {
		for (size_t i = 0; i < dlss_words.size(); ++i) {
			ShaderTextureResource descriptor {};
			std::copy_n(dlss_words[i], 8, descriptor.fields);
			if (!descriptor.IsNull()) {
				dlss_bindings[i] = resolve(dlss_words[i], false);
				BindImage(dlss_bindings[i]->image_id, false);
			}
		}
	}
	const auto acquire = [&](TextureBinding& binding, const uint32_t* words, bool storage) {
		auto& previous = cache.GetImage(binding.image_id);
		if (!previous.registered || previous.binding.needs_rebind) {
			previous.binding = {};
			binding = resolve(words, storage);
			BindImage(binding.image_id, storage);
		}
		binding.image_view = cache.FindTexture(binding.image_id, binding.desc);
	};
	for (size_t i = 0; i < dlss_words.size(); ++i) {
		if (dlss_bindings[i]) {
			acquire(*dlss_bindings[i], dlss_words[i], false);
		}
	}
	acquire(input, request.input, false);
	acquire(output, request.output, true);
	auto& source = cache.GetImage(input.image_id);
	auto& destination = cache.GetImage(output.image_id);
	const auto& source_view = input.desc.view_info;
	const auto& destination_view = output.desc.view_info;
	EXIT_IF(source.info.IsDepth() || destination.info.IsDepth() ||
	        source.backing.samples != 1 || destination.backing.samples != 1 ||
	        (input.image_id == output.image_id &&
	         source_view.base_level == destination_view.base_level &&
	         source_view.base_layer == destination_view.base_layer));

	buffer.EndRendering();
	const auto command = buffer.Handle();
	const auto range   = [](const ImageViewInfo& view) {
		return ImageSubresourceRange {view.base_level, 1, view.base_layer, 1};
	};
	source.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead, range(source_view),
	               command);
	destination.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
	                    range(destination_view), command);
	if (m_pssr_pipeline == nullptr) {
		CreatePssrPipeline();
	}
	const LinearPssr linear {m_pssr_pipeline_layout, m_pssr_pipeline, input.image_view,
	                         output.image_view, parameters};
	if (dlss_bindings[0]) {
		const auto resource = [&](const TextureBinding& binding, bool write) {
			const auto& image  = cache.GetImage(binding.image_id);
			const auto& view   = binding.desc.view_info;
			const auto  extent = binding.desc.info.MipExtent(view.base_level);
			// DLSS takes a depth image in its backing format and depth aspect.
			const bool depth  = image.info.IsDepth();
			const auto aspect = depth ? vk::ImageAspectFlagBits::eDepth : view.aspect;
			return Dlss::Resource {
			    .view       = static_cast<VkImageView>(binding.image_view),
			    .image      = static_cast<VkImage>(image.backing.image),
			    .range      = {static_cast<VkImageAspectFlags>(aspect), view.base_level, 1,
				               view.base_layer, 1},
			    .format     = static_cast<VkFormat>(depth ? image.backing.format : view.format),
			    .width      = extent.width,
			    .height     = extent.height,
			    .read_write = write};
		};
		for (const auto& binding: dlss_bindings) {
			if (binding) {
				cache.GetImage(binding->image_id)
				    .Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead,
					         range(binding->desc.view_info), command);
			}
		}
		auto exposure = dlss_bindings[2] ? resource(*dlss_bindings[2], false) : Dlss::Resource {};
		Dlss::Inputs inputs {resource(input, false), resource(*dlss_bindings[0], false),
		                     resource(*dlss_bindings[1], false),
		                     dlss_bindings[2] ? &exposure : nullptr, resource(output, true)};
		bool reset = false;
		if (auto* feature = DlssFeatureFor(request, inputs, reset); feature != nullptr) {
			// NGX records into the Vulkan command buffer, so DLSS runs where the command buffer
			// is recorded; there it falls back to the linear upscale when NGX refuses the feature
			// or the evaluation fails.
			command.Custom([feature, request = request, inputs, exposure,
			                has_exposure = dlss_bindings[2].has_value(), reset,
			                linear](vk::CommandBuffer vk_command) {
				auto replayed      = inputs;
				auto exposure_copy = exposure;
				replayed.exposure  = has_exposure ? &exposure_copy : nullptr;
				if (!EvaluateDlss(vk_command, *feature, request, replayed, reset)) {
					linear.Record(vk_command);
				}
			});
			ResetBindings();
			return;
		}
	}
	command.Custom([linear](vk::CommandBuffer vk_command) { linear.Record(vk_command); });
	ResetBindings();
}

} // namespace Libs::Graphics
