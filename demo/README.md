# Demos

1. Set the following environment variables:

- REDIS_MODULE_PATH - Relative path to compiled falkordb.so

  ```sh
  export REDIS_MODULE_PATH=<ABSOLUTE_PATH_TO_MODULE.SO>
  ```

- REDIS_PATH - Absolute path to redis-server

  ```sh
  export REDIS_PATH=<ABSOLUTE_PATH_TO_REDIS-SERVER>
  ```

2. Install requirements:

  From either <FALKORDB_ROOT>/demo/imdb or <FALKORDB_ROOT>/demo/social, Run:

  ```sh
  pip install -r requirements.txt
  ```

3. Run demo:

  ```sh
  python imdb_demo.py
  ```

  or

  ```sh
  python social_demo.py
  ```
