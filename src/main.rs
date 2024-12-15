use std::error::Error;
use std::collections::HashSet;
use std::collections::HashMap;

mod load_data;
mod similarity;
mod clusters;
mod book_recs;
mod cluster_reps;

fn main() -> Result<(), Box<dyn Error>> {
    // load data
    let ratings = load_data::load_ratings("ratings.csv")?;
    
    let mut book_ids: HashMap<u32, usize> = HashMap::new();
    for (index, rating) in ratings.iter().enumerate() {
        book_ids.entry(rating.book_id).or_insert(index);
    }

    // get # of books and # of users
    let num_books = book_ids.len();
    let num_users = ratings.iter().map(|r| r.user_id).max().unwrap() as usize + 1;

    // build rating matrix
    let rating_matrix = load_data::build_rating_matrix(&ratings, num_books, num_users);

    // compute similarity
    let similarity_matrix = similarity::compute_similarity(&rating_matrix);

    // select books reps
    let k = 5;
    let user_id = 1;
    let num_recs = 5;
    let books_recs = book_recs::recommend_books(user_id, &rating_matrix, &similarity_matrix, k, num_recs, &book_ids);

    let user_cluster_labels = clusters::k_means_clustering(&rating_matrix, k, 100);
    let reps = cluster_reps::select_rep(&rating_matrix, &user_cluster_labels, k);

    println!("Book Representatives for Each Cluster: {:?}", reps);
    println!("Recommended Books for user {:?}: {:?}", user_id, books_recs);

    Ok(())
}