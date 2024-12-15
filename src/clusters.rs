use ndarray::{Array1, Array2, ArrayView1};

pub fn k_means_clustering(data: &Array2<f32>, k: usize, max_iterations: usize) -> Vec<usize> {
    let num_points = data.nrows();
    let mut centroids = ndarray::Array2::<f32>::zeros((k, data.ncols()));
    let mut labels = vec![0; num_points];

    // initialize centroids
    for a in 0..k {
        centroids.row_mut(a).assign(&data.row(a % num_points));
    }

    // assign points to centroids
    for _ in 0..max_iterations {
        for a in 0..num_points {
            let point = data.row(a);
            let mut best_dist = f32::MAX;
            let mut best_centr = 0;
            for b in 0..k {
                let dist = (&point - &centroids.row(b)).mapv(|x| x.powi(2)).sum().sqrt();
                if dist < best_dist {
                    best_dist = dist;
                    best_centr = b;
                }
            }
            labels[a] = best_centr;
        }

        // update centroids
        for b in 0..k {
            let cluster_points: Vec<ArrayView1<f32>> = (0..num_points).filter_map(|i| {
                if labels[i] == b {
                    Some(data.row(i))
                } else {
                    None
                }
            }).collect();
        if !cluster_points.is_empty() {
            let new_centr: Array1<f32> = ndarray::Array1::from_shape_fn(data.ncols(), |col| {
                cluster_points.iter().map(|p| p[col]).sum::<f32>() / cluster_points.len() as f32
            });
            centroids.row_mut(b).assign(&new_centr);
            }
        }
    }
    return labels;
}

#[test]
fn test_k_means_clustering() {
    let data = Array2::from(vec![
        vec![1.0, 1.0],
        vec![1.0, 2.0],
        vec![4.0, 4.0],
        vec![5.0, 5.0]
    ]);

    let labels = k_means_clustering(&data, 2, 100);
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[2], labels[3]);
}
