use cblas_sys::{self, cblas_dgemm};

extern crate blas_src as _;
//extern crate openblas_src as _;

fn main() {
    let size = i32::MAX as usize + 1;
    //let size: usize = 10_000;
    println!("size: {}", size);

    println!("alloc...");
    let a: Vec<f64> = vec![1.0; size];
    println!("alloc.");
    //let b: Vec<f32> = vec![1.0; size];
    let mut c: Vec<f64> = vec![0.0; 1];

    unsafe {
        cblas_dgemm(
            cblas_sys::CBLAS_ORDER::CblasColMajor,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            1,
            1,
            size as cblas_sys::cblas_int,
            1.0,
            a.as_ptr(),
            1,
            a.as_ptr(),
            size as cblas_sys::cblas_int,
            0.0,
            c.as_mut_ptr(),
            1,
        )
    };

    println!("Result: {} vs {}", c[0], size as f64);
}
