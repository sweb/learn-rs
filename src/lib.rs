/// Implementation of CART regression / classification trees, based on Elements of
/// statistical learning (ESL).
use ndarray::prelude::*;
/// Defines a decision tree region, based on a given feature space and the associated
/// labels.
#[derive(PartialEq, Debug, Clone)]
struct Region {
    x: Array2<f64>,
    y: Array1<f64>,
    c: f64,
}

impl Region {
    /// Construct a region from feature matrix and labels represented by a 2D and 1D array.
    pub fn from_arrays(x: Array2<f64>, y: Array1<f64>) -> Self {
        if x.nrows() != y.shape()[0] {
            panic!(
                "Feature matrix ({}) and labels ({}) have different number of samples!",
                x.nrows(),
                y.shape()[0]
            )
        }
        let c = y.mean().unwrap_or(0.0);
        Self { x, y, c }
    }
    /// Number of samples of the region
    pub fn size(&self) -> usize {
        self.x.nrows()
    }

    /// Computes the prediction of samples falling into this region using the mean.
    pub fn prediction(&self) -> f64 {
        self.c
    }

    /// Partitions a region into two new regions, based on a splitting column j and a
    /// splitting point s
    pub fn partition(&self, j: usize, s: f64) -> (Region, Region) {
        let sel = self.x.index_axis(Axis(1), j);
        let (s1, s2): (Vec<(_, _)>, Vec<(_, _)>) =
            sel.indexed_iter().partition(|(_, elem)| elem <= &&s);

        let s1: Vec<usize> = s1.iter().map(|(dim, _)| *dim).collect();
        let s2: Vec<usize> = s2.iter().map(|(dim, _)| *dim).collect();

        let x1 = self.x.select(Axis(0), s1.as_slice());
        let y1 = self.y.select(Axis(0), s1.as_slice());

        let x2 = self.x.select(Axis(0), s2.as_slice());
        let y2 = self.y.select(Axis(0), s2.as_slice());

        (Region::from_arrays(x1, y1), Region::from_arrays(x2, y2))
    }

    /// Performs an exhaustive search of all possible splitting points for a given column.
    /// The splitting point with the lowest costs (squared difference between actual and
    /// prediction of both resulting regions)
    fn choose_per_column(&self, j: usize) -> Result<(f64, f64), String> {
        const PRECISION: f64 = 10_000_000.;
        let selected_column = self.x.index_axis(Axis(1), j);
        let uniques = selected_column.fold(std::collections::HashSet::new(), |mut accu, v| {
            accu.insert((*v * PRECISION) as i64);
            accu
        });

        let unique_values: Vec<f64> = uniques.into_iter().map(|x| x as f64 / PRECISION).collect();

        let mut cost_hat = f64::MAX;
        let mut s_hat = None;

        for s in unique_values {
            let (r1, r2) = self.partition(j, s);
            let c1 = r1.prediction();
            let c2 = r2.prediction();
            let cost = splitting_cost(r1.y, r2.y, c1, c2);

            if cost < cost_hat {
                cost_hat = cost;
                s_hat = Some(s);
            }
        }
        match s_hat {
            Some(s) => Ok((s, cost_hat)),
            None => Err("Could not find any possible split".to_string()),
        }
    }

    /// Chooses the best column with the best splitting point by selecting the combination
    /// with the lowest costs.
    pub fn choose_overall(&self) -> Result<SplitBoundary, String> {
        let last_col = self.x.ncols();
        let mut j_hat = None;
        let mut s_hat = None;
        let mut cost_hat = f64::MAX;
        for j in 0..last_col {
            let (s, cost) = self.choose_per_column(j)?;
            if cost < cost_hat {
                j_hat = Some(j);
                s_hat = Some(s);
                cost_hat = cost;
            }
        }
        match (j_hat, s_hat) {
            (Some(j), Some(s)) => Ok(SplitBoundary { j, s }),
            _ => Err("Could not find splitting variable and splitting point".to_string()),
        }
    }
}

/// Calculates the cost of splitting a region into two regions
fn splitting_cost(y1: Array1<f64>, y2: Array1<f64>, c1: f64, c2: f64) -> f64 {
    (y1 - c1).mapv(|x| x.powi(2)).sum() + (y2 - c2).mapv(|x| x.powi(2)).sum()
}

/// Stores decision tree splits as column/splitting point combinations
pub struct SplitBoundary {
    j: usize,
    s: f64,
}

type NodeId = usize;

/// Elements of a decision tree
struct Node {
    region: Region,
    boundary: Option<SplitBoundary>,
    left_child: Option<NodeId>,
    right_child: Option<NodeId>,
}

impl Node {
    pub fn new(region: Region) -> Self {
        Node {
            region,
            boundary: None,
            left_child: None,
            right_child: None,
        }
    }
}

/// Decision tree for regression
/// The nodes are stored in a vector and node relationships are defined by using the
/// positions of the nodes within the vector.
pub struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    pub fn new(x: Array2<f64>, y: Array1<f64>, min_node_size: usize) -> Self {
        let root = Node::new(Region::from_arrays(x, y));
        let mut result = Tree { nodes: Vec::new() };

        let root_id = result.insert_node(root);
        let mut stack = vec![root_id];

        while let Some(current_node_id) = stack.pop() {
            let current_node = &mut result.nodes[current_node_id];

            let boundary = current_node.region.choose_overall().unwrap();
            let (r1, r2) = current_node.region.partition(boundary.j, boundary.s);
            if r1.size() >= min_node_size && r2.size() >= min_node_size {
                let n1 = Node::new(r1);
                let n2 = Node::new(r2);

                let (n1_id, n2_id) = result.update_to_interior(boundary, current_node_id, n1, n2);

                stack.push(n1_id);
                stack.push(n2_id);
            }
        }
        result
    }

    fn insert_node(&mut self, n: Node) -> NodeId {
        let result = self.nodes.len();
        self.nodes.push(n);
        result
    }

    fn update_to_interior(
        &mut self,
        boundary: SplitBoundary,
        p_id: NodeId,
        n1: Node,
        n2: Node,
    ) -> (usize, usize) {
        let n1_id = self.insert_node(n1);
        let n2_id = self.insert_node(n2);

        let parent = &mut self.nodes[p_id];

        parent.boundary = Some(boundary);
        parent.left_child = Some(n1_id);
        parent.right_child = Some(n2_id);

        (n1_id, n2_id)
    }

    pub fn predict(&self, x: Array1<f64>) -> f64 {
        let mut current_node = &self.nodes[0];

        while let Some(ref boundary) = current_node.boundary {
            if x[boundary.j] <= boundary.s {
                current_node = &self.nodes[current_node.left_child.unwrap()]
            } else {
                current_node = &self.nodes[current_node.right_child.unwrap()]
            }
        }
        current_node.region.prediction()
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, Array1, Array2};
    #[test]
    fn test_region_size() {
        let r = Region::from_arrays(Array2::<f64>::zeros((2, 2)), Array1::<f64>::zeros(2));
        assert_eq!(2, r.size())
    }

    #[test]
    fn test_region_prediction() {
        let r = Region::from_arrays(Array2::<f64>::zeros((2, 2)), arr1(&[1.0, 2.0]));
        assert_eq!(1.5, r.prediction())
    }

    #[test]
    fn test_region_partition() {
        let x = arr2(&[[2., 3., 6.], [4., 5., 7.]]);
        let y = arr1(&[1., 1.]);
        let r = Region::from_arrays(x, y);
        let r1 = Region::from_arrays(arr2(&[[2., 3., 6.]]), arr1(&[1.]));
        let r2 = Region::from_arrays(arr2(&[[4., 5., 7.]]), arr1(&[1.]));

        let (act_r1, act_r2) = r.partition(0, 2.);

        assert_eq!(act_r1, r1);
        assert_eq!(act_r2, r2);
    }

    #[test]
    fn test_region_choose_per_column() {
        let features = &[[2., 3., 6.], [4., 3., 6.], [3., 3., 6.]];
        let x = arr2(features);
        let y = arr1(&[1., 0., 1.]);
        let r = Region::from_arrays(x, y);

        let (s_act, _) = r.choose_per_column(0).unwrap();

        assert_eq!(3., s_act);
    }

    #[test]
    fn test_region_choose_overall() {
        let features = &[[2., 2., 6.], [2., 4., 6.], [2., 3., 6.]];
        let x = arr2(features);
        let y = arr1(&[1., 0., 1.]);
        let r = Region::from_arrays(x, y);

        let boundary = r.choose_overall().unwrap();

        assert_eq!(1, boundary.j);
        assert_eq!(3., boundary.s);
    }

    #[test]
    fn test_splitting_cost() {
        let y1 = arr1(&[3., 2.]);
        let y2 = arr1(&[5., 9.]);
        let c1 = 2.5;
        let c2 = 7.;

        assert_eq!(0.5 + 8., splitting_cost(y1, y2, c1, c2))
    }

    #[test]
    fn test_tree_new() {
        let features = &[[2., 2., 6.], [2., 4., 6.], [2., 3., 6.]];
        let x = arr2(features);
        let y = arr1(&[0.9, 0., 1.]);

        let tree = Tree::new(x, y, 1);

        assert_eq!(5, tree.nodes.len());
        assert_eq!(1, tree.nodes[0].boundary.as_ref().unwrap().j);
        assert_eq!(
            1.9,
            tree.nodes[tree.nodes[0].left_child.unwrap()].region.y.sum()
        );
    }

    #[test]
    fn test_tree_predict() {
        let features = &[[2., 2., 6.], [2., 4., 6.], [2., 3., 6.]];
        let x = arr2(features);
        let y = arr1(&[0.9, 0., 1.]);

        let tree = Tree::new(x, y, 1);

        assert_eq!(0.9, tree.predict(arr1(&[2., 2., 6.,])));
        assert_eq!(0., tree.predict(arr1(&[2., 4., 6.,])));
        assert_eq!(1., tree.predict(arr1(&[2., 3., 6.,])));
    }
}
