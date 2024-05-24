use super::*;

/// Directed Acyclic Graph, the neural network
#[derive(Clone, Debug)]
pub struct DAC {
    pub nodes: Vec<Box<Node>>,
    order: Vec<usize>,
}

impl DAC {
    // /// The constructor for the pendulum network in [Pezzza's video](https://www.youtube.com/watch?v=EvV5Qtp_fYg)
    // fn pezzzas_pendulum() -> Self {
    //     Self {
    //         nodes: vec![
    //             Box::new(Node::new(0.0, 0.0, identity, vec![], vec![], 0)), // Cart x
    //             Box::new(Node::new(0.0, 0.0, identity, vec![], vec![], 0)), // Pendulum x
    //             Box::new(Node::new(0.0, 0.0, identity, vec![], vec![], 0)), // Pendulum y
    //             Box::new(Node::new(0.0, 0.0, identity, vec![], vec![], 0)), // Angular velocity

    //             Box::new(Node::new(0.0, 0.0, tanh, vec![], vec![], 1)),
    //         ],
    //         order: vec![],
    //     }
    // }

    pub(crate) fn new(nodes: Vec<Box<Node>>) -> Self {
        Self {
            order: Vec::with_capacity(nodes.len()),
            nodes,
        }
    }

    /// Whether any connection is active.
    pub fn has_linked(&self) -> bool {
        self.nodes.iter().any(|node| !node.children.is_empty())
    }

    /// Whether more connections could be added.
    pub fn has_unlinked(&self) -> bool {
        self.nodes.iter().any(|child_node| {
            child_node.parents.len()
                < self
                    .nodes
                    .iter()
                    .filter(|node| node.layer < child_node.layer)
                    .count()
        })
    }

    /// Sets the node processing order, such that children are always processed aftertheir parents.
    pub(crate) fn reordered(&mut self) {
        // let start = std::time::Instant::now();

        let len = self.nodes.len();
        self.order = Vec::with_capacity(len);
        let mut active_nodes: Vec<bool> = vec![true; len];

        while self.order.len() != len {
            // println!("Starting nodes: {:?}", self.nodes);
            // println!("Active nodes: {:?}", active_nodes);
            let active_nodes_clone = active_nodes.clone();
            self.nodes
                .iter()
                .enumerate()
                // Filter ---------
                .zip(active_nodes.iter_mut())
                .filter(|(_, kept)| **kept)
                // ----------------
                .for_each(|((node_i, node), kept)| {
                    if node
                        .parents
                        .iter()
                        .all(|parent| !active_nodes_clone[*parent])
                    {
                        self.order.push(node_i);
                        *kept = false;
                    }
                });
        }
        // println!("Reordered in: {}", start.elapsed().as_secs_f32());
        // println!("Reordered");
    }

    pub(crate) fn zeroed(&mut self) {
        self.nodes.iter_mut().for_each(|node| {
            node.val = 0.0;
        })
    }

    pub(crate) fn run(&mut self) {
        // let start = std::time::Instant::now();

        let mut vals: Vec<f32> = self.nodes.iter().map(|node| node.val).collect();
        for index in self.order.iter() {
            self.nodes[*index].val = vals[*index];
            self.nodes[*index].update();
            self.nodes[*index]
                .children
                .iter()
                .for_each(|(child_i, conn_weight)| {
                    vals[*child_i] += self.nodes[*index].val * conn_weight;
                })
        }

        // println!("{:?}", self.nodes);

        // println!("Ran in: {}", start.elapsed().as_secs_f32());
    }

    pub fn newnode_mutated(&self) -> Self {
        let mut new = self.clone();

        let connection_parent_i: usize;
        let connection_child_i: usize;
        let connection_weight: f32;
        let nodes_len = new.nodes.len();
        loop {
            let parents_first = fastrand::bool();
            let node_i = fastrand::usize(0..nodes_len);
            if parents_first {
                if !new.nodes[node_i].parents.is_empty() {
                    connection_child_i = node_i;
                    connection_parent_i = new.nodes[node_i].parents
                        [fastrand::usize(0..new.nodes[node_i].parents.len())];
                    connection_weight = new.nodes[connection_parent_i]
                        .children
                        .iter()
                        .filter(|child| child.0 == connection_child_i)
                        .collect::<Vec<&(usize, f32)>>()[0]
                        .1;
                    break;
                } else {
                    if !new.nodes[node_i].children.is_empty() {
                        connection_parent_i = node_i;
                        (connection_child_i, connection_weight) = new.nodes[node_i].children
                            [fastrand::usize(0..new.nodes[node_i].children.len())];
                        break;
                    }
                }
            } else {
                if !new.nodes[node_i].children.is_empty() {
                    connection_parent_i = node_i;
                    (connection_child_i, connection_weight) = new.nodes[node_i].children
                        [fastrand::usize(0..new.nodes[node_i].children.len())];
                    break;
                } else {
                    if !new.nodes[node_i].children.is_empty() {
                        connection_child_i = node_i;
                        connection_parent_i = new.nodes[node_i].parents
                            [fastrand::usize(0..new.nodes[node_i].parents.len())];
                        connection_weight = new.nodes[connection_parent_i]
                            .children
                            .iter()
                            .filter(|child| child.0 == connection_child_i)
                            .collect::<Vec<&(usize, f32)>>()[0]
                            .1;
                        break;
                    }
                }
            }
        }

        // Update parent
        let parent_inner_i = new.nodes[connection_child_i]
            .parents
            .iter()
            .position(|&parent| parent == connection_parent_i)
            .unwrap();
        new.nodes[connection_child_i].parents.remove(parent_inner_i);
        new.nodes[connection_child_i].parents.push(nodes_len);

        // Update child
        let child_inner_i = new.nodes[connection_parent_i]
            .children
            .iter()
            .position(|&(child, _)| child == connection_child_i)
            .unwrap();
        new.nodes[connection_parent_i]
            .children
            .remove(child_inner_i);
        new.nodes[connection_parent_i]
            .children
            .push((nodes_len, connection_weight));

        let mut update_layers = false;
        let new_node_layer: u32 =
            if new.nodes[connection_child_i].layer - new.nodes[connection_parent_i].layer == 1 {
                update_layers = true;
                new.nodes[connection_child_i].layer
            } else {
                fastrand::u32(
                    (new.nodes[connection_parent_i].layer + 1)
                        ..(new.nodes[connection_child_i].layer),
                )
            };

        if update_layers {
            new.nodes
                .iter_mut()
                .filter(|node| node.layer >= new_node_layer)
                .for_each(|node| node.layer += 1);
        }

        new.nodes.push(Box::new(Node::new(
            0.0,
            0.0,
            *NEWNODE_ACTIVATION_F.get().unwrap(),
            vec![connection_parent_i],
            vec![(connection_child_i, connection_weight)],
            new_node_layer,
        )));
        new
    }

    pub fn newconnection_mutated(&self) -> Self {
        let mut new = self.clone();

        let mut node_a_i: usize;
        let node_b_i: usize;
        loop {
            node_a_i = fastrand::usize(0..new.nodes.len());
            let children: Vec<usize> = new.nodes[node_a_i]
                .children
                .iter()
                .map(|child| child.0)
                .collect();
            let parents: Vec<usize> = new.nodes[node_a_i].parents.clone();
            let possible_node_b: Vec<usize> = (0..new.nodes.len())
                .filter(|index| {
                    (new.nodes[*index].layer > new.nodes[node_a_i].layer
                        && !children.contains(index))
                        || (new.nodes[*index].layer < new.nodes[node_a_i].layer
                            && !parents.contains(index))
                })
                .collect();
            match possible_node_b.is_empty() {
                true => {
                    continue;
                }
                false => {
                    node_b_i = possible_node_b[fastrand::usize(0..possible_node_b.len())];
                    break;
                }
            }
        }

        let a_is_child = new.nodes[node_a_i].layer > new.nodes[node_b_i].layer;
        if a_is_child {
            new.nodes[node_a_i].parents.push(node_b_i);
            new.nodes[node_b_i].children.push((node_a_i, 0.5));
        } else {
            new.nodes[node_b_i].parents.push(node_a_i);
            new.nodes[node_a_i].children.push((node_b_i, 0.5));
        }

        new
    }

    pub fn weight_mutated(&self) -> Self {
        let mut new = self.clone();

        let children_count = new
            .nodes
            .iter()
            .flat_map(|node| node.children.iter())
            .count();
        let index = fastrand::usize(0..children_count);
        new.nodes
            .iter_mut()
            .flat_map(|node| node.children.iter_mut())
            .nth(index)
            .unwrap()
            .1 += fastrand::f32() * 2.0 - 1.0;

        new
    }

    pub fn bias_mutated(&self) -> Self {
        let mut new = self.clone();
        let index = fastrand::usize(0..new.nodes.len());
        new.nodes[index].bias += fastrand::f32() - 0.5;
        new
    }

    pub fn unmutated(&self) -> Self {
        self.clone()
    }
}
