use csv;
use std::error::Error;
use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct Rating {
    pub user_id: u32,
    pub book_id: u32,
    pub rating: f32,
}

pub fn load_ratings(file_path: &str) -> Result<Vec<Rating>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(file_path)?;
    let mut ratings = Vec::new();

    for result in reader.records() {
        let record = result?;
        let user_id: u32 = record[0].parse()?;
        let book_id: u32 = record[1].parse()?;
        let rating: f32 = record[2].parse()?;
        ratings.push(Rating {user_id, book_id, rating});
    }
    Ok(ratings)
}

pub fn build_rating_matrix(ratings: &[Rating], num_books: usize, num_users: usize) -> ndarray::Array2<f32> {
    let mut matrix = Array2::<f32>::zeros((num_users, num_books));
    for rating in ratings {
        let user_id = rating.user_id;
        let book_id = rating.book_id;
        let rating = rating.rating;
        
        if (user_id as usize) < num_users && (book_id as usize) < num_books{
            matrix[(user_id as usize, book_id as usize)] = rating;
        }
    }
    return matrix
}