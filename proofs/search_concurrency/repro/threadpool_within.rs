// Repro for the two shutdown_within bugs (proofs/search_concurrency PoolWithin.lean).
// Run: new cargo crate (edition 2024; crossfire =3.1.19, once_cell =1.21.4, parking_lot =0.13.0),
// copy graph/src/threadpool.rs to src/threadpool.rs and this file to src/main.rs; cargo run.
// Output at main ca36bfa15: A: first=false second=true ; B: Err(Timeout) after 3 s for a 50 ms limit.

#[allow(dead_code)]
mod threadpool;
use std::{sync::{Arc, mpsc}, thread, time::{Duration, Instant}};
use threadpool::ThreadPool;

fn blocked_pool() -> (Arc<ThreadPool>, mpsc::Sender<()>) {
    let pool = Arc::new(ThreadPool::new(1));
    let (release, blocked) = mpsc::channel::<()>();
    let (started, has_started) = mpsc::channel();
    pool.spawn(move || { started.send(()).unwrap(); let _ = blocked.recv(); }, None);
    has_started.recv().unwrap();
    (pool, release)
}
fn main() {
    // BUG A: second shutdown_within reports true while the worker is still running
    let (pool, release) = blocked_pool();
    let t = Instant::now();
    let first = pool.shutdown_within(Duration::from_millis(50));
    let second = pool.shutdown_within(Duration::from_millis(50));
    println!("A: first={first} second={second} (worker still blocked) elapsed={:?}", t.elapsed());
    drop(release);
    // BUG B: concurrent blocking shutdown() holds the lock; shutdown_within(50ms) overshoots
    let (pool, release) = blocked_pool();
    let p1 = Arc::clone(&pool);
    thread::spawn(move || p1.shutdown());
    thread::sleep(Duration::from_millis(200));
    let (done, rx) = mpsc::channel();
    let p2 = Arc::clone(&pool);
    let t = Instant::now();
    thread::spawn(move || { done.send(p2.shutdown_within(Duration::from_millis(50))).unwrap(); });
    let r = rx.recv_timeout(Duration::from_secs(3));
    println!("B: shutdown_within(50ms) result after {:?}: {:?}", t.elapsed(), r);
    drop(release);
    let r = rx.recv_timeout(Duration::from_secs(3));
    println!("B: after releasing the job: {:?}", r);
}
