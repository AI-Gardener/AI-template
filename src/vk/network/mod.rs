use std::{collections::HashMap, fs::File, io::BufReader, path::Path};

use fastrand::f32;

use crate::InputVertex;

// Code from a vulkanalia tutorial: https://github.com/KyleMayes/vulkanalia/blob/master/tutorial/src/27_model_loading.rs
/// Loads the models from an OBJ file as one mesh.
pub(crate) fn load_models<P: AsRef<Path>>(path: P) -> (Vec<InputVertex>, Vec<u32>) {
    // Model

    let mut reader = BufReader::new(File::open(path).unwrap());

    let (models, _) = tobj::load_obj_buf(
        &mut reader,
        &tobj::LoadOptions {
            single_index: false,
            triangulate: true,
            ignore_points: true,
            ignore_lines: true,
        },
        |_| Ok(Default::default()),
    ).unwrap();

    // Vertices / Indices

    let mut unique_vertices: HashMap<InputVertex, u32> = HashMap::new();

    let (mut vertices, mut indices): (Vec<InputVertex>, Vec<u32>) = (Vec::new(), Vec::new());
    vertices.extend([
        InputVertex::new(
            [0.05, 0.0, 0.1, 1.0],
            [-2.0, -1.0, 0.5],
            0
        ),
        InputVertex::new(
            [0.05, 0.0, 0.1, 1.0],
            [-2.0, 1.0, 0.5],
            0
        ),
        InputVertex::new(
            [0.05, 0.0, 0.1, 1.0],
            [2.0, -1.0, 0.5],
            0
        ),
        InputVertex::new(
            [0.05, 0.0, 0.1, 1.0],
            [2.0, 1.0, 0.5],
            0
        ),
    ]);
    indices.extend([
        0, 1, 2,
        3, 2, 1,
    ]);

    let mut z: f32 = -0.1;
    for model in &models {
        z+=0.1;
        for index in &model.mesh.indices {
            let pos_offset = (3 * index) as usize;

            let vertex: InputVertex = 
            if -model.mesh.positions[pos_offset + 2] < 0.1 {
                InputVertex::new(
                    [(f32() + 4.0) / 8.0, (f32() + 1.0) / 8.0, (f32() + 1.0) / 8.0, 1.0],
                    [
                        (model.mesh.positions[pos_offset]) / 2.0,
                        (- model.mesh.positions[pos_offset + 2]) / 2.0,
                        z,
                    ],
                    1,
                )
            } else if -model.mesh.positions[pos_offset + 2] < 0.3 {
                InputVertex::new(
                    [0.0, (f32() + 4.0) / 8.0, 0.0, 1.0],
                    [
                        (model.mesh.positions[pos_offset]) / 2.0,
                        (- model.mesh.positions[pos_offset + 2] - 0.2) / 2.0,
                        z,
                    ],
                    2,
                )
            } else {
                InputVertex::new(
                    [1.0, 1.0, 1.0, 1.0],
                    [
                        (model.mesh.positions[pos_offset]) / 2.0,
                        (- model.mesh.positions[pos_offset + 2]) / 2.0,
                        z,
                    ],
                    3,
                )
                    
            };

            if let Some(index) = unique_vertices.get(&vertex) {
                indices.push(*index);
            } else {
                let index = vertices.len() as u32;
                unique_vertices.insert(vertex, index);
                vertices.push(vertex);
                indices.push(index);
            }
        }
    }

    (vertices, indices)
}