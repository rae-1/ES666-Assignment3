import pdb
import glob
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import time
# from src.JohnDoe import some_function
# from src.JohnDoe.some_folder import folder_func


class data():
    indexA: int
    indexB: int
    distance: float



class PanaromaStitcher():
    turn: bool
    def __init__(self):
        turn = True

    def make_panaroma_for_images_in(self, image1, image2, showMatches=False):
        # image1_path, image2_path = 'Images/I1/STA_0031.JPG', 'Images/I1/STB_0032.JPG'
        # image1 = cv2.imread(image1_path)
        # image2 = cv2.imread(image2_path)
        # print(type(image1), type(image2))
        # image1 = cv2.resize(image1, (image1.shape[0]//3, image1.shape[1]//4))
        # image2 = cv2.resize(image2, (image2.shape[0]//3, image2.shape[1]//4))

        # Detects the keypoints and extracts the local invariant descriptors from the images
        kpsA, featuresA = self.keypoint_and_descriptors(image1)
        kpsB, featuresB = self.keypoint_and_descriptors(image2)

        # draw_keypoints = cv2.drawKeypoints(image2, kpsB, 0, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # plt.imshow(cv2.cvtColor(draw_keypoints, cv2.COLOR_BGR2RGB))
        # plt.show()

        good_matches = self.match_keypoints(kpsA, kpsB, featuresA, featuresB)
        if good_matches is None:
            print("less then 4 matched featuers")
            return None

        homography_mat = self.apply_ransac(good_matches, kpsA, kpsB)
        print(homography_mat)
        rows1, cols1 = image1.shape[:2]
        rows2, cols2 = image2.shape[:2]

        points1 =  np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
        points  =  np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
        # points2 =  cv2.perspectiveTransform(points, homography_mat)
        points2 = self.custom_perspective_transform(points, homography_mat)
        list_of_points = np.concatenate((points1,points2), axis=0)

        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

        H_translation = np.array([
                                    [1, 0, (-x_min)],
                                    [0, 1, (-y_min)],
                                    [0, 0,     1   ]
                                ]).dot(homography_mat)

        # output_img = cv2.warpPerspective(image1, H_translation, (x_max-x_min, y_max-y_min))
        output_img = self.custom_warp_perspective(image1, H_translation, (x_max-x_min, y_max-y_min))
        print(output_img.shape)
        # output_img[(-y_min):rows2+(-y_min), (-x_min):cols2+(-x_min)] = image2
        output_img[(-y_min):(-y_min)+rows2, (-x_min):(-x_min) + cols2] = self.overlay_images(output_img[(-y_min):(-y_min) + rows2, (-x_min):(-x_min) + cols2], image2)
        output_img = self.crop_black_borders(output_img)
        result_img = output_img

        # plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        # plt.show()

        return result_img

        # return result
        # # return stitched_image, homography_matrix_list


    def crop_black_borders(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Taking into account the largest contour
            # and cropping the image to reduce the number of pixels and the bounding box
            x, y, w, h = cv2.boundingRect(contours[0])
            cropped_image = image[y:y+h, x:x+w]
            return cropped_image
        return image


    def overlay_images(self, base_img, overlay_img):
        # To handle broadcasting errors while adding the other half with the 2nd image
        rows, cols, _ = base_img.shape
        overlay_img_resized = cv2.resize(overlay_img, (cols, rows))
        return overlay_img_resized


    def keypoint_and_descriptors(self,image):
        print("keypoints detection and extracting descriptors")
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        (kps, descriptors) = sift.detectAndCompute(gray_img, None)
        return (kps, descriptors)


    def match_keypoints(self,kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4.0):
        print("Finding the matches")
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        # list of 2 matches for each feature each member of the list is a tuple of 2 matches
        # each match is a tuple of 3 values: trainIdx, queryIdx, and distance
        # queryIdx is the index of the feature in the first image and trainIdx is the index of the feature in the second image
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        good_matches = []    # Initializing actual matches list

        # Lowes ratio test for computing good matches
        for match in rawMatches:
            if match[0].distance < match[1].distance * ratio:
                good_matches.append((match[0].queryIdx, match[0].trainIdx))

        # For computing the homography matrix 4 matches needed
        if len(good_matches) > 4:
            return good_matches

        return None


    def apply_ransac(self, good_points, kpsA, kpsB, iterations=2000, threshold=5):
        '''
        randomly select 4 points, compute the homoraphy matrix, check how many points
        align with the other points on the other plane within some threshold and
        repeat the process for iter iterations for finding the best H.
        '''
        print("Proceeding to find the homography matrix")
        max_inliers = []
        H_mat = []
        for i in range(iterations):
            # Selecting 4 random points
            selected_points = random.sample(good_points, 4)
            H = self.compute_homography(selected_points, kpsA, kpsB)
            cur_inliers = []
            for point in good_points:
                x1, y1 = kpsA[point[0]].pt
                x2, y2 = kpsB[point[1]].pt
                # Converting it to homogenous coordinates
                pt1 = np.array([x1, y1, 1])
                pt2 = np.array([x2, y2, 1])
                # Taking the projection and checking for within threshold
                proj_point = H @ pt1
                proj_point = proj_point / proj_point[2]     # Need to normalize the point
                if np.linalg.norm(pt2 - proj_point) < threshold:
                    cur_inliers.append(point)
            if len(cur_inliers) > len(max_inliers):
                max_inliers = cur_inliers
                H_mat = H

        return H_mat


    def compute_homography(self, selected_points, kpsA, kpsB):
        A = []
        # Stacking the A matrix
        for pt in selected_points:
            x1, y1 = kpsA[pt[0]].pt
            x2, y2 = kpsB[pt[1]].pt
            A.append([x1, y1, 1, 0, 0, 0, -1*x2*x1, -1*x2*y1, -1*x2])
            A.append([0, 0, 0, x1, y1, 1, -1*y2*x1, -1*y2*y1, -1*y2])

        A = np.array(A)
        U, S, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)
        H = H / H[2][2]     # DOF 8 need to make sure the last element is 1
        return H

    def custom_perspective_transform(self, points, homography_mat):
        num_points = points.shape[0]
        homogeneous_points = np.hstack([points.reshape(num_points, 2), np.ones((num_points, 1))])
        transformed_points = homography_mat @ homogeneous_points.T

        # Converting back to Cartesian coordinates
        transformed_points /= transformed_points[2, :]
        transformed_points = transformed_points[:2, :].T
        transformed_points = transformed_points.reshape(num_points, 1, 2)

        return transformed_points


    def custom_warp_perspective(self, image, homography_mat, output_size):
        height, width = output_size[1], output_size[0]
        output_img = np.zeros((height, width, image.shape[2]), dtype=image.dtype)

        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        homogeneous_coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones_like(x_coords).ravel()])

        inv_homography_mat = np.linalg.inv(homography_mat)
        transformed_coords = inv_homography_mat @ homogeneous_coords
        transformed_coords /= transformed_coords[2, :]

        x_transformed = transformed_coords[0, :].reshape(height, width).astype(np.float32)
        y_transformed = transformed_coords[1, :].reshape(height, width).astype(np.float32)

        # Interpolating pixel values
        for i in range(height):
            for j in range(width):
                x, y = x_transformed[i, j], y_transformed[i, j]
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    output_img[i, j] = self.bilinear_interpolate(image, x, y)

        return output_img

    def bilinear_interpolate(self, image, x, y):
        x0 = int(np.floor(x))
        x1 = min(x0 + 1, image.shape[1] - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, image.shape[0] - 1)

        Ia = image[y0, x0]
        Ib = image[y1, x0]
        Ic = image[y0, x1]
        Id = image[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id


    # def make_panaroma_for_images_in(self, path='./Images/I1'):
    #     path = path + '/*'
    #     image_paths = glob.glob(path)
    #     image_paths.sort()
    #     images = [cv2.imread(impath) for impath in image_paths]
    #     images = [cv2.resize(image, (450, 450)) for image in images]

    #     num_images = len(images)
    #     stitcher = PanaromaStitcher()

    #     mid_image_index = num_images // 2
    #     left, right = mid_image_index - 1, mid_image_index + 1
    #     cur_image = images[mid_image_index]
    #     turn = True
    #     while left >= 0 or right < num_images:
    #         try:
    #             if turn:
    #                 stitched_image = stitcher.make_panaroma_for_images_in(cur_image, images[right])
    #                 right += 1
    #                 turn = False
    #             else:
    #                 stitched_image = stitcher.make_panaroma_for_images_in(images[left], cur_image)
    #                 turn = True
    #                 left -= 1

    #             if stitched_image is None:
    #                 print("Less then 4 features were detected")
    #                 break
    #             print(left, right)
    #             cur_image = stitched_image
    #         except Exception as e:
    #             turn = not turn
    #             print("ERROR: ", left, right)

    #     plt.imshow(cur_image)
    #     plt.show()
    #     result_path = f'results/{pth.split("/")[-1]}.png'
    #     cv2.imwrite(result_path, cur_image)
    #     print(f"Panorama saved to {result_path}")
    #     # break

all_submissions = glob.glob('./Images/*')
for pth in all_submissions:
    path = pth + '/*'
    image_paths = glob.glob(path)
    image_paths.sort()

    images = [cv2.imread(impath) for impath in image_paths]
    images = [cv2.resize(image, (350, 350)) for image in images]

    num_images = len(images)
    stitcher = PanaromaStitcher()

    mid_image_index = num_images // 2
    left, right = mid_image_index - 1, mid_image_index + 1
    cur_image = images[mid_image_index]
    turn = True

    while left >= 0 or right < num_images:
        try:
            if turn:
                stitched_image = stitcher.make_panaroma_for_images_in(cur_image, images[right])
                right += 1
                turn = False
            else:
                stitched_image = stitcher.make_panaroma_for_images_in(images[left], cur_image)
                turn = True
                left -= 1

            if stitched_image is None:
                print("Less then 4 features were detected")
                break
            print(left, right)
            cur_image = stitched_image
            print("cur_image:", type(cur_image))
            # plt.imshow(cur_image)
            # plt.show()
        except Exception as e:
            print(e)
            turn = not turn
            print("cur_image in except:", type(cur_image))
            print("ERROR: ", left, right)
            time.sleep(5)

    plt.imshow(cur_image)
    plt.show()
    result_path = f'results/{pth.split("/")[-1]}.png'
    cv2.imwrite(result_path, cur_image)
    print(f"Panorama saved to {result_path}")


# PanaromaStitcher().make_panaroma_for_images_in('Images/I1')