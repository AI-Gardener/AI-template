use serde::{Deserialize, Serialize};

use crate::{ActivationFunction, Node};

#[derive(Deserialize, Serialize)]
struct Point {
    pub(crate) bias: f32,
    pub(crate) activation_f_id: u32,
    pub(crate) parents: Vec<usize>,
    pub(crate) children: Vec<(usize, f32)>,
    pub(crate) layer: u32,
}

/// An intermediary struct used for deserializing/serializing `DAC` data
#[derive(Deserialize, Serialize)]
pub struct Network(Vec<Point>);

impl Network {
    pub(crate) fn from_nodes(nodes: &Vec<Box<Node>>) -> Self {
        Self(nodes.iter().map(|node| Point {
            bias: node.bias,
            activation_f_id: node.activation_function as u32,
            parents: node.parents.clone(),
            children: node.children.clone(),
            layer: node.layer,
        }).collect())
    }

    pub(crate) fn to_nodes(self) -> Vec<Box<Node>> {
        self.0.into_iter().map(|point| Box::new(Node::new(
            point.bias,
            ActivationFunction::from_id(point.activation_f_id),
            point.parents,
            point.children,
            point.layer,
        ))).collect()
    }
}