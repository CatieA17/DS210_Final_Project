use std::collections::HashSet;
use std::collections::HashMap;
use ndarray::ArrayView1;
use crate::clusters;

pub fn recommend_books(user_id: usize, rating_matrix: &ndarray::Array2<f32>, similarity_matrix: &ndarray::Array2<f32>, k: usize, num_recs: usize, book_ids: &HashMap<u32, usize>) -> Vec<u32> {
    let user_ratings = rating_matrix.row(user_id);
    let num_users = rating_matrix.nrows();
    let num_books = rating_matrix.ncols();

    // k-means clustering based on user's ratings
    let user_cluster_labels = clusters::k_means_clustering(rating_matrix, k, 100);
    let user_cluster = user_cluster_labels[user_id];

    let similar_users: Vec<usize> = user_cluster_labels.iter().enumerate().filter(|&(_, label)| *label == user_cluster).map(|(index, _)| index).collect();
    
    let mut book_scores: Vec<(u32, f32)> = Vec::new();

    // calc unweighted score of books not rated by the user
    for book_id in 0..num_books {
        if user_ratings[book_id] != 0.0 {
            continue;
        }
            let mut score = 0.0;
        
            // calc score based on similarity on unrated books vs. rated books
            for &similar_user in &similar_users {
                if similar_user == user_id {
                    continue;
                }
                if rating_matrix[[similar_user, book_id]] > 0.0 {
                    score += rating_matrix[[similar_user, book_id]] * similarity_matrix[[user_id, similar_user]];
                }
            }
            if let Some((&book_id, _)) = book_ids.iter().find(|&(_, &idx)| idx == book_id) {
                book_scores.push((book_id, score));
            }
        }
    book_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let recomended_books: Vec<u32> = book_scores.iter().filter(|(book_id, _)| user_ratings[*book_ids.get(book_id).unwrap()] == 0.0).take(num_recs).map(|(book_id, _)| *book_id).collect();
    return recomended_books
}