from sift import sift_feature_fetcher as sift
import aiofiles, asyncio
from functools import partial
import numpy
import cv2


def load_training_file():
    with open("images/octopus/files.txt") as f:
        file_dict = dict(item.strip().split(" ") for item in f.readlines())
        return file_dict


def convert_image_to_df(raw_data, label):
    npimage = numpy.asarray(bytearray(raw_data), dtype=numpy.uint8)
    img = cv2.imdecode(npimage, -1)
    _, kp, features = sift.fetch_sift_info(img, sift.to_gray)
    return features, label


async def load_file_data(filename, label, loop, thread_pool):
    try:
        async with aiofiles.open(f"images/octopus/training/{filename}", "rb") as image:
            print(f'Opening: "{image._file.name}"')
            return await loop.run_in_executor(thread_pool, convert_image_to_df, await image.read(), label)
    except Exception as e:
        print(e)
        return None


class Training:
    thread_number = 10

    def generate_training_data(self):
        from concurrent import futures
        tasks = []
        file_dict = load_training_file()

        loop = asyncio.get_event_loop()
        with futures.ThreadPoolExecutor(self.thread_number) as executor:
            __load_file_data = partial(load_file_data, loop=loop, thread_pool=executor)
            for filename, label in file_dict.items():
                tasks.append(asyncio.ensure_future(__load_file_data(filename, label)))

            loop.run_until_complete(asyncio.wait(tasks))

        for task in tasks:
            print(task._state, task.result())

        loop.close()
