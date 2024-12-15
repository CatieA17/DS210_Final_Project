use ndarray::prelude::*;

// select book closest to centroid
pub fn select_rep(data: &ndarray::Array2<f32>, labels:&[usize], k:usize) -> Vec<usize> {
    let mut reps = vec![0; k];
    for cluster_id in 0..k {
        let cluster_points: Vec<usize> = labels.iter().enumerate().filter_map(| (i, &label)| 
        if label == cluster_id {
            Some(i)
        } else {
            None
        }).collect();

        let centr = data.select(ndarray::Axis(0), &cluster_points).mean_axis(ndarray::Axis(0)).unwrap();

        let mut min_dist = f32::MAX;
        let mut rep = 0;
        for &i in &cluster_points {
            let dist = (&data.row(i) - &centr).mapv(|x| x.powi(2)).sum().sqrt();
            if dist < min_dist {
                min_dist == dist;
                rep = i;
            }
        }
        reps[cluster_id] = rep;
    }
    return reps
}
