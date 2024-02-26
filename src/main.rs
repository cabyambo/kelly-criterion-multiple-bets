use std::vec;

use itertools::Itertools;
use nalgebra::DVector;

fn main() {
    println!("Hello, world!");

    // Probability bet succeeds, win multiplier if bet does succeed.
    let bets_data: Vec<(f64,f64)> = vec![
        (0.3, 12.8), 
        (0.4, 6.4),
        (0.5, 3.2),
        (0.6, 1.6),
        (0.7, 0.8),
        (0.8, 0.4),
        (0.9, 0.2)];

    let bets_data_defined_loss: Vec<(f64,f64, f64)> = vec![
        (0.3, 12.8, 0.5), 
        (0.4, 6.4, 0.5),
        (0.5, 3.2, 0.5),
        (0.6, 1.6, 0.5),
        (0.7, 0.8, 0.5),
        (0.8, 0.4, 0.5),
        (0.9, 0.2, 0.5)];
    
    let alpha = 0.01;
    let max_iter = 10000;

    let (value_var, fs_var) = multiple_simultaneous_kelly(&bets_data, alpha, max_iter);
    let (value_var_defined, fs_var_defined) = multiple_simultaneous_kelly_defined_loss(&bets_data_defined_loss, alpha, max_iter);

    println!("Optimized value of the objective function: {}", value_var);
    println!("Optimal fractions of wealth to be wagered on each bet: {:?}", fs_var);

    println!("Optimized value of the objective function with defined losses: {}", value_var_defined);
    println!("Optimal fractions of wealth to be wagered on each bet with defined losses: {:?}", fs_var_defined);
}


pub fn multiple_simultanous_expectation_log_wealth(bets_data: &Vec<(f64,f64)>, fs: &Vec<f64>) -> (f64, Vec<f64>) {
    assert_eq!( bets_data.len(), fs.len() );
    let mut res = 0f64;
    let mut grad = DVector::from_element(bets_data.len(), 0f64);
    for indices in std::iter::repeat(0).take(bets_data.len()).map(|i| i..i+2).multi_cartesian_product() {
        let mut prob = 1f64;
        let mut wealth = 1f64;
        let mut local_grad = DVector::from_element(bets_data.len(), 0f64);
        for (i, &j) in indices.iter().enumerate() {
            if j == 0 {
                prob *= bets_data[i].0;
                wealth += fs[i] * bets_data[i].1;
                local_grad[i] += bets_data[i].1;
            } else {
                prob *= 1f64 - bets_data[i].0;
                wealth -= fs[i];
                local_grad[i] -= 1f64;
            }
        }
        if wealth > 0f64 {
            res += prob * wealth.ln();
            grad += &(prob * &local_grad / wealth);
        }
    }
    (res, grad.as_slice().into())
}

pub fn clip(x: f64) -> f64 {
    if x > 1f64 {
        1f64
    } else if x < 0f64 {
        0f64
    } else {
        x
    }
}

pub fn multiple_simultaneous_kelly(bets_data: &Vec<(f64,f64)>, alpha: f64, max_iter: usize) -> (f64, Vec<f64>) {
    let mut value_var = 0f64;
    let mut fs_var = vec![0f64; bets_data.len()];
    for _ in 0..max_iter {
        let (value, grad) = multiple_simultanous_expectation_log_wealth( &bets_data, &fs_var );
        let mat = DVector::from_row_slice(fs_var.as_slice()) + DVector::from_row_slice(grad.as_slice()) * alpha;
        let mut fs_candidate = mat.iter()
            .map(|&x| clip(x))
            .collect::<Vec<f64>>();
        // Scale weights if total greater than 1
        let total : f64 = fs_candidate.iter().sum();
        if total > 1f64 {
            for x in fs_candidate.iter_mut() {
                *x = *x / total;
            };
        }
        // Stop if no improvement in expectation
        let (value_candidate, _) = multiple_simultanous_expectation_log_wealth( &bets_data, &fs_candidate);
        if value_candidate <= value {
            break;
        } else {
            fs_var = fs_candidate;
            value_var = value_candidate;
        }
    }
    (value_var, fs_var)
}


// MARK: -- Predefined loss
pub fn multiple_simultanous_expectation_log_wealth_defined_loss(bets_data: &Vec<(f64, f64, f64)>, fs: &Vec<f64>) -> (f64, Vec<f64>) {
    assert_eq!( bets_data.len(), fs.len() );
    let mut res = 0f64;
    let mut grad = DVector::from_element(bets_data.len(), 0f64);
    for indices in std::iter::repeat(0).take(bets_data.len()).map(|i| i..i+2).multi_cartesian_product() {
        let mut prob = 1f64;
        let mut wealth = 1f64;
        let mut local_grad = DVector::from_element(bets_data.len(), 0f64);
        for (i, &j) in indices.iter().enumerate() {
            if j == 0 {
                prob *= bets_data[i].0;
                wealth += fs[i] * bets_data[i].1;
                local_grad[i] += bets_data[i].1;
            } else {
                prob *= 1f64 - bets_data[i].0;
                wealth -= fs[i] * bets_data[i].2;
                local_grad[i] -= 1f64;
            }
        }
        if wealth > 0f64 {
            res += prob * wealth.ln();
            grad += &(prob * &local_grad / wealth);
        }
    }
    (res, grad.as_slice().into())
}

pub fn multiple_simultaneous_kelly_defined_loss(bets_data: &Vec<(f64,f64, f64)>, alpha: f64, max_iter: usize) -> (f64, Vec<f64>) {
    let mut value_var = 0f64;
    let mut fs_var = vec![0f64; bets_data.len()];
    for _ in 0..max_iter {
        let (value, grad) = multiple_simultanous_expectation_log_wealth_defined_loss( &bets_data, &fs_var );
        let mat = DVector::from_row_slice(fs_var.as_slice()) + DVector::from_row_slice(grad.as_slice()) * alpha;
        let mut fs_candidate = mat.iter()
            .map(|&x| clip(x))
            .collect::<Vec<f64>>();
        // Scale weights if total greater than 1
        let total : f64 = fs_candidate.iter().sum();
        if total > 1f64 {
            for x in fs_candidate.iter_mut() {
                *x = *x / total;
            };
        }
        // Stop if no improvement in expectation
        let (value_candidate, _) = multiple_simultanous_expectation_log_wealth_defined_loss( &bets_data, &fs_candidate);
        if value_candidate <= value {
            break;
        } else {
            fs_var = fs_candidate;
            value_var = value_candidate;
        }
    }
    (value_var, fs_var)
}