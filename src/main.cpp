#include "window.hpp"
#include "shared.inl"
#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/task_graph.hpp>

int main(int argc, char const *argv[])
{
    // Create a window
    auto window = AppWindow("Learn Daxa", 860, 640);

    daxa::Instance instance = daxa::create_instance({});

    daxa::Device device = instance.create_device_2(instance.choose_device({}, {}));

    auto pipeline_manager = daxa::PipelineManager({
        .device = device,
        .shader_compile_options = {
            .root_paths = {
                DAXA_SHADER_INCLUDE_DIR,
                "src/",
            },
            .language = daxa::ShaderLanguage::SLANG,
            .enable_debug_info = true,
        },
        .name = "my pipeline manager",
    });

    // clang-format off
    std::shared_ptr<daxa::ComputePipeline> compute_pipeline;
    {
        auto result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"compute.slang"}, 
                .compile_options = {
                    .entry_point = "entry_compute_shader",
                },
            },
            .push_constant_size = sizeof(ComputePush),
            .name = "my pipeline",
        });
        if (result.is_err())
        {
            std::cerr << result.message() << std::endl;
            return -1;
        }
        compute_pipeline = result.value();
    }

    daxa::BufferId histogram_buffer = device.create_buffer(daxa::BufferInfo{
        .size = sizeof(u32) * 256,
        .name = "histogram_buffer",
    });

    daxa::TaskBuffer task_histogram_buffer{{.initial_buffers = {.buffers = std::array{histogram_buffer}}, .name = "histogram_buffer"}};

    daxa::TaskGraph task_graph = daxa::TaskGraph({
        .device = device,
        .name = "my task graph",
    });
    {
        task_graph.use_persistent_buffer(task_histogram_buffer);

        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE, task_histogram_buffer),
            },
            .task = [compute_pipeline, histogram_buffer](daxa::TaskInterface ti)
            {
                auto p = ComputePush{
                    .global_histograms = ti.device.device_address(histogram_buffer).value(),
                };
                ti.recorder.set_pipeline(*compute_pipeline);
                ti.recorder.push_constant(p);
                ti.recorder.dispatch({.x = 1, .y = 1, .z = 1});
            },
            .name = ("compute_histogram"),
        });
        task_graph.submit({});
        task_graph.complete({});
    };

    while (!window.should_close()){
        window.update();
        pipeline_manager.reload_all();

        // So, now all we need to do is execute our task graph!
        task_graph.execute({});
        device.wait_idle();
        device.collect_garbage();
    }

    return 0;
}