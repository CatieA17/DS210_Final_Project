use ndarray::prelude::*;
use ndarray::{Array2, ArrayView1};

// computes cosine similarity
pub fn cosine_similarity(v1: &ArrayView1<f32>, v2: &ArrayView1<f32>) -> f32 {
    let dot_product = v1.dot(v2);
    let norm_v1 = v1.dot(v1).sqrt();
    let norm_v2 = v2.dot(v2).sqrt();
    return dot_product/(norm_v1 * norm_v2)
}

// computes similarity matrix
pub fn compute_similarity(rating_matrix: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    let num_books = rating_matrix.ncols();
    let mut similarity_matrix = ndarray::Array2::<f32>::zeros((num_books, num_books));

    for a in 0..num_books {
        for b in a..num_books {
            let book1 = rating_matrix.column(a);
            let book2 = rating_matrix.column(b);
            let similarity = cosine_similarity(&book1, &book2);
            similarity_matrix[[a, b]] = similarity;
            similarity_matrix[[b, a]] = similarity;
        }
    }
    return similarity_matrix
}

#[test]
fn test_compute_similarity_matrix() {
    let matrix = Array2::from(vec![
        vec![5.0, 3.0],
        vec![4.0, 0.0],
    ]);
    let similarity = compute_similarity(&matrix);
    assert!((similarity[[0, 1]] - 0.9746318461970762).abs() < 1e-6, "similarity matrix test failed");
}