## Running

Make sure to have three gRPC workers at the right endpoints before running these examples. Moose offers the option to run gRPC workers with in-memory storage or with file storage (accept numpy and csv files).

### With In-memory Storage

Before runnning `grpc_with_in_memory_storage.py` launch three Comet workers as follows (run each command in its own terminal):

```sh
$ cargo run --bin comet -- --identity localhost:50000 --port 50000
$ cargo run --bin comet -- --identity localhost:50001 --port 50001
$ cargo run --bin comet -- --identity localhost:50002 --port 50002
```

### With File Storage

Before runnning `grpc_with_filesystem_storage.py` launch three Comet workers as follows (run each command in its own terminal):

```sh
$ cargo run --bin comet -- --identity localhost:50000 --port 50000 --file-system-storage
$ cargo run --bin comet -- --identity localhost:50001 --port 50001 --file-system-storage
$ cargo run --bin comet -- --identity localhost:50002 --port 50002 --file-system-storage
```