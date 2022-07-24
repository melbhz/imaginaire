import os
import math
import time
import numpy as np
from shutil import copyfile
import multiprocessing as mp
import cv2


def save_zoom_up_images(FROM_DIR, TO_DIR, IMGSIZE = 256 * 2, NUM_CPUS = 24):
    # FROM_DIR = '/data/scratch/projects/punim1358/Datasets/NSW_SA2/AnxietyDiag/train/images_a'
    # TO_DIR = '/data/scratch/projects/punim1358/Datasets/NSW_SA2/AnxietyDiag/train_imagesa_zoom20'
    # IMGSIZE = 256 * 2
    # NUM_CPUS = 24

    if not os.path.exists(FROM_DIR):
        print("{} not exist!".format(FROM_DIR))
        return
    if not os.path.exists(TO_DIR):
        print("{} not exist, creating it...".format(TO_DIR))
        os.makedirs(TO_DIR)

    dirList = os.listdir(TO_DIR)
    downloadedImagesDict = {}
    start = time.time()
    for image in dirList:
        downloadedImagesDict[image] = image
    print(f'already saved images: {len(downloadedImagesDict)}')
    print(f'time last: {time.time() - start}')

    start = time.time()
    onlyfiles = os.listdir(FROM_DIR)
    # os.listdir(FROM_DIR) is fast but not as safe as using [f for f in os.listdir(FROM_DIR) if os.path.isfile(os.path.join(FROM_DIR, f))]
    print(f'from dir images: {len(onlyfiles)}')
    print(f'time last: {time.time() - start}')

    start = time.time()
    fn_dict = {}
    filename_list = []
    todo_dict = {}
    todo_set = set()
    for i in range(len(onlyfiles)):
        filename = onlyfiles[i]

        if (i + 1) % 10000 == 0:
            print(f'in loop time last {i + 1}: {time.time() - start}        {len(filename_list)}')
            start = time.time()

        coordinate = filename.strip().split('~')[1]
        fn_dict[coordinate] = filename
        x, y = [int(x) for x in coordinate.split(',')]
        y_zoomup = math.floor(y / 2)
        x_zoomup = math.floor(x / 2)
        newfilename = str(x_zoomup) + ',' + str(y_zoomup) + '.jpg'

        # if newfilename in filename_list or newfilename in downloadedImagesDict:#####
        if newfilename in todo_set or newfilename in downloadedImagesDict:  #####
            continue
        else:
            filename_list.append(newfilename)
            todo_dict[newfilename] = 1
            todo_set.add(newfilename)

    print(f'images to get: {len(filename_list)}')

    '''
    #retrieve images using parallel processing
    cluster = Pool(NUM_THREADS)
    cluster.starmap(merge_image, zip(count_list, filename_list))
    cluster.close()
    cluster.join()    
    '''

    num_point_per_cpu = math.ceil(len(filename_list) / NUM_CPUS)
    filename_list_nested = [filename_list[i * num_point_per_cpu:(i + 1) * num_point_per_cpu]
                            if (i + 1) * num_point_per_cpu < len(filename_list)
                            else filename_list[i * num_point_per_cpu:]
                            for i in range(NUM_CPUS)]
    if len(filename_list_nested) != NUM_CPUS:
        print("!!!!!!!!!!!!!!!!!!!!! len(filename_list_nested) != NUM_CPUS")
        return

    pool = mp.Pool()
    for i in range(NUM_CPUS):
        pool.apply_async(merge_image_list, args=(i, filename_list_nested[i], fn_dict, FROM_DIR, TO_DIR, IMGSIZE))

    '''    
    results = []
    for i in range(len(filename_list)):
        results.append(pool.apply_async(merge_image, args=(i,filename_list[i])))
    '''

    pool.close()
    pool.join()
    print("zoom up conversion finished!")


def merge_image_list(cpuid, filename_list, fn_dict, FROM_DIR, TO_DIR, IMGSIZE, TEST=False):
    # FROM_DIR = '/data/scratch/projects/punim1358/nearmap_images/WA/test'
    # TO_DIR = '/data/scratch/projects/punim1358/HZ/nearmap_images/WA/zm20/test'
    # IMGSIZE= 256

    print(f'cpuid: {cpuid}')
    print(f'len(filename_list): {len(filename_list)}')

    start = time.time()
    for idx in range(len(filename_list)):
        filename = filename_list[idx]
        if (idx + 1) % 1000 == 0:
            print(f'cpu {cpuid} - time last {idx + 1}: {time.time() - start}       of {len(filename_list)}')
            start = time.time()

        coordinate, _ = filename.strip().split('.')
        x, y = [int(x) for x in coordinate.split(',')]
        mini_tiles = [(0, 0), (1, 0), (0, 1), (1, 1)]
        mini_tiles = [(x * 2 + i, y * 2 + j) for (i, j) in mini_tiles]
        mini_tiles = [str(xx) + ',' + str(yy) for (xx, yy) in mini_tiles]

        imgs_all_exist = True
        for xy in mini_tiles:
            if not xy in fn_dict:
                imgs_all_exist = False
                break
        if imgs_all_exist:
            img_paths = [os.path.join(FROM_DIR, fn_dict[yx]) for yx in mini_tiles]
            for img in img_paths:
                if not os.path.isfile(img):
                    imgs_all_exist = False
                    print('impossible: check for error!!')
                    break
        if imgs_all_exist:
            inputImages = []
            for img in img_paths:
                image = cv2.imread(img)  # Image.open(img)
                inputImages.append(image)
            # OutImg = cv2.vconcat(cv2.hconcat([cv2.imread(img_paths[0]), cv2.imread(img_paths[1])]), cv2.hconcat([cv2.imread(img_paths[2]), cv2.imread(img_paths[3])]))
            hconcat_up = cv2.hconcat([inputImages[0], inputImages[1]])
            hconcat_down = cv2.hconcat([inputImages[2], inputImages[3]])
            OutImg = cv2.vconcat([hconcat_up, hconcat_down])
            OutImg = cv2.resize(OutImg, (IMGSIZE, IMGSIZE))
            cv2.imwrite(os.path.join(TO_DIR, filename), OutImg)

            # cv2.imshow(newfilename, OutImg)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if TEST:
                print("TEST!")
                TEST = False
                for img in img_paths:
                    print(f'copy images from {img} to {os.path.join(".", os.path.basename(img))}')
                    copyfile(img, os.path.join('.', os.path.basename(img)))
                print(f'copy images from {os.path.join(TO_DIR, filename)} to {os.path.join(".", filename)}')
                copyfile(os.path.join(TO_DIR, filename), os.path.join('.', filename))

    print(f'cpu {cpuid} - done.')


def save_zoom_up_large_images(FROM_DIR, TO_DIR, N_XY = 5, IMGSIZE = 256, NUM_CPUS = 24):
    # FROM_DIR = '/data/scratch/projects/punim1358/Datasets/NSW_SA2/AnxietyDiag/train/images_b'
    # TO_DIR = '/data/scratch/projects/punim1358/Datasets/NSW_SA2/AnxietyDiag/train_imagesb_n10'
    # N_XY = 10
    # IMGSIZE = 256 * N_XY
    # NUM_CPUS = 24
    N_STEP = N_XY

    if not os.path.exists(FROM_DIR):
        print("{} not exist!".format(FROM_DIR))
        return
    if not os.path.exists(TO_DIR):
        print("{} not exist, creating it...".format(TO_DIR))
        os.makedirs(TO_DIR)

    dirList = os.listdir(TO_DIR)
    downloadedImagesDict = {}
    start = time.time()
    for image in dirList:
        downloadedImagesDict[image] = image
    print(f'already saved images: {len(downloadedImagesDict)}')
    print(f'time last: {time.time() - start}')

    start = time.time()
    onlyfiles = os.listdir(FROM_DIR)
    # os.listdir(FROM_DIR) is fast but not as safe as using [f for f in os.listdir(FROM_DIR) if os.path.isfile(os.path.join(FROM_DIR, f))]
    print(f'from dir images: {len(onlyfiles)}')
    print(f'time last: {time.time() - start}')

    fn_dict = {filename.strip().split('~')[1]: filename for filename in onlyfiles}
    fn_array = np.array(list(fn_dict.items()))
    print(f'fn_array.shape: {fn_array.shape}')

    coords = [[int(x) for x in coord.split(',')] for coord in fn_dict.keys()]
    coord_array = np.array(coords) #, dtype=np.uint32)
    print(f'coord_array.shape: {coord_array.shape}')
    print(f'coord_array.dtype: {coord_array.dtype}')
    print(f'coord_array[:5] {coord_array[:5]}')

    minXY = np.amin(coord_array[:, :2], axis=0)
    maxXY = np.amax(coord_array[:, :2], axis=0)
    # minXY = np.amin(data[:, :2].astype(np.int), axis=0)
    # maxXY = np.amax(data[:, :2].astype(np.int), axis=0)
    print('minXY, maxXY = {}, {}'.format(minXY, maxXY))
    # return


    start = time.time()
    filename_list = []

    for x in range(minXY[0], maxXY[0], N_STEP):
        for y in range(minXY[1], maxXY[1], N_STEP):
            newfilename = str(x) + ',' + str(y) + '.jpg'
            xy = str(x) + ',' + str(y)
            if newfilename in downloadedImagesDict or xy not in fn_dict:
                continue
            else:
                filename_list.append(newfilename)

    print(f'images to get: {len(filename_list)}')

    num_point_per_cpu = math.ceil(len(filename_list) / NUM_CPUS)
    filename_list_nested = [filename_list[i * num_point_per_cpu:(i + 1) * num_point_per_cpu]
                            if (i + 1) * num_point_per_cpu < len(filename_list)
                            else filename_list[i * num_point_per_cpu:]
                            for i in range(NUM_CPUS)]
    if len(filename_list_nested) != NUM_CPUS:
        print("!!!!!!!!!!!!!!!!!!!!! len(filename_list_nested) != NUM_CPUS")
        return

    pool = mp.Pool()
    for i in range(NUM_CPUS):
        pool.apply_async(merge_image_list_large, args=(i, filename_list_nested[i], fn_dict, FROM_DIR, TO_DIR, IMGSIZE, N_XY))

    '''    
    results = []
    for i in range(len(filename_list)):
        results.append(pool.apply_async(merge_image, args=(i,filename_list[i])))
    '''

    pool.close()
    pool.join()
    print("zoom up conversion finished!")


def merge_image_list_large(cpuid, filename_list, fn_dict, FROM_DIR, TO_DIR, IMGSIZE, N_XY=10):
    print(f'cpuid: {cpuid}')
    print(f'len(filename_list): {len(filename_list)}')
    cols = rows = N_XY

    start = time.time()
    for idx in range(len(filename_list)):
        filename = filename_list[idx]
        if (idx + 1) % 1000 == 0:
            print(f'cpu {cpuid} - time last {idx + 1}: {time.time() - start}       of {len(filename_list)}')
            start = time.time()

        coordinate, _ = filename.strip().split('.')
        x, y = [int(x) for x in coordinate.split(',')]
        mini_tiles = [['{},{}'.format(x + i, y + j) for i in range(cols)] for j in range(rows)]

        imgs_all_exist = True
        for xy_list in mini_tiles:
            for xy in xy_list:
                if not xy in fn_dict:
                    imgs_all_exist = False
                    break
            if not imgs_all_exist:
                break

        if imgs_all_exist:
            im_list_2d = [[cv2.imread(os.path.join(FROM_DIR, fn_dict[xy])) for xy in xy_list] for xy_list in mini_tiles]
            OutImg = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
            OutImg = cv2.resize(OutImg, (IMGSIZE, IMGSIZE))
            cv2.imwrite(os.path.join(TO_DIR, filename), OutImg)

    print(f'cpu {cpuid} - done.')



def save_combined_tile_images(FROM_DIR, TO_DIR, N_XY = 5, IMGSIZE = 256, NUM_CPUS = 24):
    # FROM_DIR = '/data/scratch/projects/punim1358/Datasets/NSW_SA2/AnxietyDiag/train/images_b'
    # TO_DIR = '/data/scratch/projects/punim1358/Datasets/NSW_SA2/AnxietyDiag/train_imagesb_n10'
    # N_XY = 10
    # IMGSIZE = 256 * N_XY
    # NUM_CPUS = 24
    N_STEP = N_XY

    if not os.path.exists(FROM_DIR):
        print("{} not exist!".format(FROM_DIR))
        return
    if not os.path.exists(TO_DIR):
        print("{} not exist, creating it...".format(TO_DIR))
        os.makedirs(TO_DIR)

    dirList = os.listdir(TO_DIR)
    downloadedImagesDict = {}
    start = time.time()
    for image in dirList:
        downloadedImagesDict[image] = image
    print(f'already saved images: {len(downloadedImagesDict)}')
    print(f'time last: {time.time() - start}')

    start = time.time()
    onlyfiles = os.listdir(FROM_DIR)
    # os.listdir(FROM_DIR) is fast but not as safe as using [f for f in os.listdir(FROM_DIR) if os.path.isfile(os.path.join(FROM_DIR, f))]
    print(f'from dir images: {len(onlyfiles)}')
    print(f'time last: {time.time() - start}')

    fn_dict = {filename.strip().split('~')[1]: filename for filename in onlyfiles}
    fn_array = np.array(list(fn_dict.items()))
    print(f'fn_array.shape: {fn_array.shape}')

    coords = [[int(x) for x in coord.split(',')] for coord in fn_dict.keys()]
    coord_array = np.array(coords) #, dtype=np.uint32)
    print(f'coord_array.shape: {coord_array.shape}')
    print(f'coord_array.dtype: {coord_array.dtype}')
    print(f'coord_array[:5] {coord_array[:5]}')

    minXY = np.amin(coord_array[:, :2], axis=0)
    maxXY = np.amax(coord_array[:, :2], axis=0)
    # minXY = np.amin(data[:, :2].astype(np.int), axis=0)
    # maxXY = np.amax(data[:, :2].astype(np.int), axis=0)
    print('minXY, maxXY = {}, {}'.format(minXY, maxXY))
    # return


    start = time.time()
    filename_list = []

    for x in range(minXY[0], maxXY[0], N_STEP):
        for y in range(minXY[1], maxXY[1], N_STEP):
            newfilename = str(x) + ',' + str(y) + '.jpg'
            xy = str(x) + ',' + str(y)
            if newfilename in downloadedImagesDict or xy not in fn_dict:
                continue
            else:
                filename_list.append(newfilename)

    print(f'images to get: {len(filename_list)}')

    num_point_per_cpu = math.ceil(len(filename_list) / NUM_CPUS)
    filename_list_nested = [filename_list[i * num_point_per_cpu:(i + 1) * num_point_per_cpu]
                            if (i + 1) * num_point_per_cpu < len(filename_list)
                            else filename_list[i * num_point_per_cpu:]
                            for i in range(NUM_CPUS)]
    if len(filename_list_nested) != NUM_CPUS:
        print("!!!!!!!!!!!!!!!!!!!!! len(filename_list_nested) != NUM_CPUS")
        return

    pool = mp.Pool()
    for i in range(NUM_CPUS):
        pool.apply_async(merge_image_list_tile, args=(i, filename_list_nested[i], fn_dict, FROM_DIR, TO_DIR, IMGSIZE, N_XY))

    '''    
    results = []
    for i in range(len(filename_list)):
        results.append(pool.apply_async(merge_image, args=(i,filename_list[i])))
    '''

    pool.close()
    pool.join()
    print("zoom up conversion finished!")


def merge_image_list_tile(cpuid, filename_list, fn_dict, FROM_DIR, TO_DIR, IMGSIZE, N_XY=10):
    print(f'cpuid: {cpuid}')
    print(f'len(filename_list): {len(filename_list)}')
    cols = rows = N_XY

    start = time.time()
    for idx in range(len(filename_list)):
        filename = filename_list[idx]
        if (idx + 1) % 1000 == 0:
            print(f'cpu {cpuid} - time last {idx + 1}: {time.time() - start}       of {len(filename_list)}')
            start = time.time()

        coordinate, _ = filename.strip().split('.')
        x, y = [int(x) for x in coordinate.split(',')]
        mini_tiles = [['{},{}'.format(x + i, y + j) for i in range(cols)] for j in range(rows)]

        imgs_all_exist = True
        for xy_list in mini_tiles:
            for xy in xy_list:
                if not xy in fn_dict:
                    imgs_all_exist = False
                    break
            if not imgs_all_exist:
                break

        if imgs_all_exist:
            im_list_2d = [[cv2.imread(os.path.join(FROM_DIR, fn_dict[xy])) for xy in xy_list] for xy_list in mini_tiles]
            OutImg = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
            OutImg = cv2.resize(OutImg, (IMGSIZE, IMGSIZE))
            cv2.imwrite(os.path.join(TO_DIR, filename), OutImg)

    print(f'cpu {cpuid} - done.')


if __name__ == "__main__":
    # save_zoom_up_images()
    # save_zoom_up_large_images()
    FROM_DIR = '/data/scratch/projects/punim1358/Datasets/NSW_SA2/AlcoholDrinksPerWeek/train/images_a'
    TO_DIR = '/data/scratch/projects/punim1358/Datasets/NSW_SA2/AlcoholDrinksPerWeek/train/images_a_n5'
    save_zoom_up_large_images(FROM_DIR, TO_DIR, N_XY=5, IMGSIZE=256, NUM_CPUS=24)
