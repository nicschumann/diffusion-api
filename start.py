import multiprocessing
import uvicorn


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    uvicorn.run('server:app', port=8000, host="0.0.0.0")
