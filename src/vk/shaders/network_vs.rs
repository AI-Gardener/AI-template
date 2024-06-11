vulkano_shaders::shader! {
    ty: "vertex",
    src: r"
        #version 450

        layout(set = 0, binding = 0) uniform ModelTransformations {
            mat4 data[32];
        } tf;
        layout(push_constant) uniform GeneralInfo {
            uvec2 resolution;
            float time;
            float hw_ratio;
            uint swap_redblue;
        } gen;

        layout(location = 0) in vec4 color;
        layout(location = 1) in vec3 position;
        layout(location = 2) in int transform_id;

        layout(location = 0) out vec4 v_color;
        layout(location = 1) out uint v_transform_id;

        void main() {
            // Vertex
            gl_Position = vec4(position, 1.0);
            // Model
            if (transform_id > -1) {
                gl_Position *= tf.data[transform_id];
            }
            // View
            gl_Position.x *= gen.hw_ratio;

            v_color = color;
            v_transform_id = transform_id;
        }
    ",
}