import time
import sys
import os
import cv2
import random
import textwrap
import subprocess
from colorsys import rgb_to_hls, hls_to_rgb
from PIL import Image
from PIL import ImageOps
from PIL import ImageFont
from PIL import ImageDraw

import tumblr_utils

'''
utils for popart image style processing
# http://miguelventura.pt/image-manipulation-with-hsl.html
'''


def rgb2hls(t):
    """ convert PIL-like RGB tuple (0 .. 255) to colorsys-like
    HSL tuple (0.0 .. 1.0) """
    r, g, b = t
    r /= 255.0
    g /= 255.0
    b /= 255.0
    return rgb_to_hls(r, g, b)


def hls2rgb(t):
    """ convert a colorsys-like HSL tuple (0.0 .. 1.0) to a
    PIL-like RGB tuple (0 .. 255) """
    r, g, b = hls_to_rgb(*t)
    r *= 255
    g *= 255
    b *= 255
    return (int(r), int(g), int(b))

''''''

'''
Camera stuff
Here should decide wether using webcam, DSLR camera or CRASH the code
'''

def shoot_webcam_pic(save_path):
    cam = cv2.VideoCapture(0)  # 0 -> index of camera
    s, img = cam.read()
    if s:  # frame captured without any errors
        cv2.imwrite(save_path, img)  # save image
    else:
        save_path = None
    return save_path


camera_shot = shoot_webcam_pic

''''''

'''
GLOBALS
'''

import globals
RANDOM_TITLES = globals.RANDOM_TITLES
print(RANDOM_TITLES)

'''
various utils
'''


def get_rand_int(rangelen=(0, len(RANDOM_TITLES) - 1)):
    return random.randint(*rangelen)


def get_timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat, end='\r')
        time.sleep(1)
        t -= 1

''''''


'''
Settings
'''

settings = {
    "__default__": {
        "padding": 20,  # px
        "gap": 10,  # px
    }
}

settings["__default__"]["gap"] = settings["__default__"]["padding"] // 2
# settings["__default__"]["padding_bottom"] = settings["__default__"]["padding"] // 2


def get_setting(set_key="padding", key="__default__"):
    # global settings
    return \
        settings \
        .get(key, {}) \
        .get(set_key, "padding")


# xoff = settings.PADDING + x * (settings.TILE_SIZE[0] + settings.GAP)
# yoff = settings.PADDING + y * (settings.TILE_SIZE[1] + settings.GAP)

CSIZE_4IMG_L = {
    "height": 1200,
    "width": 1800
}
CSIZE_4IMG_P = {}
# swap for portrait
CSIZE_4IMG_P["height"], CSIZE_4IMG_P["width"] = \
    CSIZE_4IMG_L["width"], CSIZE_4IMG_L["height"]


im_padding = get_setting("padding")
im_gap = get_setting("gap")

imgs_height = CSIZE_4IMG_L["height"] - (2 * im_padding + 1 * im_gap)
imgs_width = CSIZE_4IMG_L["width"]



def create_pic_folder(name_sufix):
    timestamp = get_timestamp()
    folder_name = "{ts}_{sfx}".format(ts=timestamp, sfx=name_sufix)
    abspath_dir = os.path.abspath(folder_name)
    os.makedirs(name=abspath_dir, exist_ok=True)
    return abspath_dir


def shoot_pics(where, number=1):
    pics = []
    countdown(1)
    for i in range(number):
        pic_name = "{ts}_{n}.png".format(ts=get_timestamp(), n=i)
        pic_path = os.path.join(where, pic_name)
        # myimg = 'img/trumpsmile1.jpg'
        # myimg = 'img/pollen-cocktail.jpg'
        # myimg = 'img/YIAC0042.jpg'
        print("SMILE ! Next picture in 3 seconds !")
        countdown(1)
        # initialize the camera
        cam = cv2.VideoCapture(0)   # 0 -> index of camera
        s, img = cam.read()
        if s:    # frame captured without any errors
            cv2.imwrite(pic_path, img)  #save image

        ########################################
        # here copy image, later get from camera
        # shutil.copyfile(myimg, pic_path)
        # capture_picture()
        ########################################
        print("SUPER !!")
        pics.append(pic_path)
    return pics


def save_pictures(number=1, name_sufix="pic"):
    save_dir = create_pic_folder(name_sufix)
    pics_path = shoot_pics(save_dir, number)
    return pics_path



def get_pic_abspath(save_folder, num):
    pic_name = "{ts}_{n}.png".format(ts=get_timestamp(), n=num)
    pic_path = os.path.join(save_folder, pic_name)
    return pic_path


def take_pics(number=1, name_sufix="pic"):
    pictures_paths = set()
    print("# {n} picture{s} going to be shot !"
          .format(n=number, s=" is" if number < 2 else "s are"))
    save_dir = create_pic_folder(name_sufix)
    print("# Saving original{s} in {dir}"
          .format(s="" if number < 2 else "s", dir=save_dir))
    print("")
    print("# Start shooting")
    for n in range(number):
        pic_abspath = get_pic_abspath(save_dir, n + 1)
        print("# Picture {n}/{total}".format(n=n + 1, total=number))
        pic_abspath = camera_shot(pic_abspath)
        pictures_paths.add(pic_abspath)
    return pictures_paths


# https://stackoverflow.com/questions/35438802/making-a-collage-in-pil
def create_collage(canvas, ncol, nrow, listofimages, xpad, ypad, title):
    i = 0
    x = xpad
    y = 0
    for col in range(ncols):
        x += xpad
        for row in range(nrows):
            y += ypad
            # y_wspace
            print(i, x, y)
            canvas.paste(listofimages[i], (x, y))
            i += 1
            y += img_height + ypad
        x += img_width + xpad
        y = 0
        print("--- after:: ", i, x, y)

        # draw.text((2 * xpad, ih + xpad), title, (0, 0, 0), font=font)
    canvas.save("Collage_" + str(get_timestamp()) + ".jpg")



"""
pop: 1 picture, no title, squared (but on normal sized canvas)
pola: 1 picture, title, normal sized canvas
gif: 4 pictures, no title, normal sized canvas
"""


def chose_title(titles=[]):
    tit = ""
    if titles:
        tit = titles[get_rand_int()]
    else:
        title_ok = False
        while title_ok is False:
            tit = catch_booth_input("What's the title :D ? KEEP IT SHORT")
            if not tit:
                pass
                # get random
            elif len(tit) > 140:
                print("Your title is even too long to tweet ahahaha")
            else:
                title_ok = True
    return tit


# acquire_photo


BOOTH = {
    'POLA': {
        'title': chose_title,
    },
    'GIF': {
        'title': chose_title,
    },
    'POP': {
        'title': chose_title,
    }
}
BOOTH_TYPES = BOOTH.keys()


def get_input(msg):
    try:
        inp = input(msg)
    except Exception as inpexc:
        sys.exit('GO FUCK YOURSELF :D')
    else:
        return inp


def catch_booth_input(msg):
    booth_type = ""  # warhol, pola, gif, or none !
    booth_type = input(msg)
    if booth_type == "": booth_type = "RANDOM"
    booth_type = booth_type.upper()
    if booth_type not in BOOTH_TYPES: return None
    return booth_type



# def resize_pola(image, ncols, nrows, canvas_width, canvas_height, **kwargs):
#     # image = resize_normal(image, ncols, nrows, max_width, max_height, **kwargs)

#     print("-- RESIZE NORMAL -- ", ncols, nrows)
#     x_wspace = kwargs.get("x_wspace", 0)
#     y_wspace = kwargs.get("y_wspace", 0)

#     max_width = int((int(canvas_width // ncols) - x_wspace))
#     max_height = int((int(canvas_height // ncols) - y_wspace))

#     # if (max_width < canvas_width

#     max_width = int(max_width + (max_width * 0.25))
#     max_height = int(max_height + (max_height * 0.25))
#     # if 4K take the middle
#     # else
#     print("max_width :: ", max_width)
#     print("max_height :: ", max_height)
#     # max_canvas_side = max(max_width, max_height)

#     widest = max(max_width, max_height)
#     rsize = (widest, widest)
#     print("Thumb Before: ", image)

#     image.thumbnail(rsize)
#     print("Thumb After: ", image)

#     print("-- RESIZE POLA -- ")

#     image_w = image.size[0]
#     image_h = image.size[1]

#     if (image_w < canvas_width) or (image_h < canvas_height):  # too small
#         image = ImageOps.fit(image, (canvas_width, canvas_height), Image.ANTIALIAS)

#     image_w = image.size[0]
#     image_h = image.size[1]

#     crop_top = 0.05 * canvas_width
#     crop_bot = 0.15 * canvas_height

#     crop_left = (0.025 * canvas_width) // 2
#     crop_right = crop_left

#     print("Before: ", image)
#     image = image.crop((crop_left, crop_top,
#                         canvas_width - crop_right, canvas_height - crop_bot))
#     print("After: ", image)
#     return image




# def resize_normal(image, ncols, nrows, max_width, max_height, **kwargs):
#     print("-- RESIZE NORMAL -- ", ncols, nrows)
#     x_wspace = kwargs.get("x_wspace", 0)
#     y_wspace = kwargs.get("y_wspace", 0)

#     max_width = int((int(max_width // ncols) - x_wspace))
#     max_height = int((int(max_height // ncols) - y_wspace))
#     # if 4K take the middle
#     # else
#     print("max_width :: ", max_width)
#     print("max_height :: ", max_height)
#     # max_canvas_side = max(max_width, max_height)
#     rsize = (max_width, max_height)
#     # ok
#     if max_width < max_height:
#         rsize = (max_height + 1, max_height)

#     image.thumbnail(rsize)
#     return image


def resize_quarter(image):
    image.thumbnail(size)
    print(im.size)


def switch_orientation(canvas_wh):
    return (canvas_wh[1], canvas_wh[0])


# from https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "

    while True:
        qutext = "{q} {p}".format(q=question, p=prompt)
        choice = input(qutext).lower().strip()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' "
                  "(or 'y' or 'n').\n")


def query_choices(question, choices=[]):
    choices = [c.lower().strip() for c in choices]
    prompt = "[ {} ]".format(" OR ".join(choices))
    qutext = "{q} {p} ".format(q=question, p=prompt)
    inp = None
    while True:
        choice = input(qutext).lower().strip()
        if (choice == '') or (choice in choices):
            return choice
        else:
            print("Possible values are: {pv}".format(pv=prompt))


def compute_whitespace(canvas_margin1, canvas_margin2,
                       n_img, img_padding1, img_padding2):
    return                                  \
        (canvas_margin1 + canvas_margin2) + \
        n_img * (img_padding1 + img_padding2)




default_booth = 'pola'

CANVAS_DIMENSIONS = (1800, 1200)

# CSIZE_4IMG_L = {
#     "height": 1200,
#     "width": 1800
# }


class Collage():
    def __init__(self):
        print("COLLAGE")
        self.name = "collage_{ts}.jpg".format(ts=get_timestamp())
        self.canvas_width = 0
        self.canvas_height = 0
        # self.canvas = Image.new('RGB',
        #                         (self.canvas_width, self.canvas_height),
        #                         (255, 255, 255))

        self.images_path = []
        self.images_objs = []

        self.pics_gaps = 0
        self.ncols = 1
        self.nrows = 1
        self.pic_number = self.ncols * self.nrows
        self.pic_shot = 1
        self.title = ""
        self.orientation = 0  # landscape

        # padding on each picture
        # This might not reflect the final padding which might be
        # recomputed to better center pictures
        self.padding_top = 10  # px
        self.padding_right = 10  # px
        self.padding_bottom = 10  # px
        self.padding_left = 10  # px

        # compute_max_whitespace
        self.x_wsp = 100
        self.y_wsp = 100
        self.max_img_width = 100
        self.max_img_height = 100
        print(vars(self))


    def compute_max_whitespace(self):
        self.x_wsp = \
            compute_whitespace(
                self.margin_left, self.margin_right,
                self.ncols, self.padding_left, self.padding_right
            )
        self.y_wsp = \
            compute_whitespace(
                self.margin_top, self.margin_bottom,
                self.nrows, self.padding_top, self.padding_bottom
            )

    def compute_max_image_size(self):
        self.max_img_width = \
            int((int(self.canvas_width // self.ncols) - self.x_wsp))
        self.max_img_height = \
            int((int(self.canvas_height // self.nrows) - self.y_wsp))

    def compute_real_padding_values(self):
        # recompute the real padding values after resizing pictures
        # this computes gaps of an equal size space wheter left middle
        # or right padding
        print("self.images_objs", self.images_objs)
        resized_width, resized_height = self.images_objs[0].size
        self.horizontal_padding = \
            int(((self.canvas_width -
                  (self.ncols * resized_width)) / (self.ncols + 1)) / 2)
        self.vertical_padding = \
            int(((self.canvas_height -
                  (self.nrows * resized_height)) / (self.nrows + 1)) / 2)
        return self.horizontal_padding, self.vertical_padding


    def thumbnailise(self, image, pick_side=min):
        '''
        Thumbnailise the picture, that is redimension keeping the aspect ratio
        By default thumbnail will resize considering the smallest side.
        When doing a polaroid, instead we want the thumbnail to size according
        to the widest side and then crop the picture to size, so the pick_side
        function becomes max() instead of min
        '''
        min_thumb_side = pick_side(self.max_img_width, self.max_img_height)
        thumb_size = (min_thumb_side, min_thumb_side)
        image.thumbnail(thumb_size)
        return image

    # https://stackoverflow.com/questions/35438802/making-a-collage-in-pil
    def make_collage(self):
        self.preprocess()
        h_padding, v_padding = self.compute_real_padding_values()
        self.make_it(h_padding, v_padding)
        self.postprocess()
        print("Saving {name}".format(name=self.name))
        self.canvas.save(self.name)
        return self.name


    def make_it(self, h_padding, v_padding):
        img_w, img_h = self.images_objs[0].size
        i = 0
        x = h_padding
        y = 0
        for col in range(self.ncols):
            x += h_padding
            for row in range(self.nrows):
                y += v_padding
                # y_wspace
                print(i, x, y)
                self.canvas.paste(self.images_objs[i], (x, y))
                i += 1
                y += img_h + v_padding
            x += img_w + h_padding
            y = 0
            print("--- after:: ", i, x, y)

    def preprocess(self):
        print("pre process Collage")
        '''
        Original images modifications
        '''
        # self.images_objs = SOMETHING_NEW
        self.resize_images()
        return

    def postprocess(self):
        print("post process Collage")
        '''
        Final canvas modifications
        '''
        # by default do nothing
        # see child class
        # self.canvas = SOMETHING_NEW
        return


class PolaroidCollage(Collage):
    def __init__(self):
        print("POLA")
        print(vars(self))
        super().__init__()
        print(vars(self))
        # this is landscape so width == width
        self.canvas_width = CANVAS_DIMENSIONS[0]
        self.canvas_height = CANVAS_DIMENSIONS[1]
        self.compute_canvas_margins()
        self.canvas = Image.new('RGB',
                                (self.canvas_width, self.canvas_height),
                                (255, 255, 255))

        # pola specific
        self.ncols = 1
        self.nrows = 1
        self.pic_number = self.ncols * self.nrows
        self.pic_shot = 1

        # Some space at bottom for title
        # approximately 15% of the canvas height
        self.margin_bottom = \
            int(0.10 * self.canvas_height) or self.margin_bottom
        # default padding values are enough
        # self.compute_max_whitespace()

        # compute stuff
        self.compute_max_whitespace()
        self.compute_max_image_size()
        self.title = chose_title(titles=RANDOM_TITLES)
        print(vars(self))

    def compute_canvas_margins(self):
        # margin canvas
        self.margin_top = 0.05 * self.canvas_height  # 5%
        self.margin_right = 0.05 * self.canvas_width  # 5%
        self.margin_bottom = 0.05 * self.canvas_height  # 5%
        self.margin_left = 0.05 * self.canvas_width  # 5%

    def resize_images(self):
        self.images_objs = [
            self.fit(thumb_image)
            for thumb_image in [
                self.thumbnailise(image, max)
                for image in self.images_objs
            ]
        ]
        # return image

    def crop(self, image):
        # crop_top = 0.05 * self.canvas_width
        # crop_bot = 0.15 * self.canvas_height
        # crop_left = (0.025 * canvas_width) // 2
        # crop_right = crop_left
        # basically create margin by cropping
        crop_left = self.margin_left
        crop_right = self.canvas_width - self.margin_right
        crop_top = self.margin_top
        crop_bot = self.canvas_height - self.margin_bottom
        image = image.crop((crop_left, crop_top, crop_right, crop_bot))
        return image

    def fit(self, image):
        print("# Computing fit of {img}".format(img=image))
        image_w = image.size[0]
        image_h = image.size[1]
        # if image is smaller, just a fit else crop
        if (image_w < self.canvas_width) or (image_h < self.canvas_height):
            print("# Expanding image to fit {img}".format(img=image))
            image = ImageOps.fit(image,
                                 (self.max_img_width, self.max_img_height),
                                 Image.ANTIALIAS)
            print("# Expanding done")
        else:
            print("# Cropping image to fit {img}".format(img=image))
            image = self.crop(image)
            print("# Cropping done")
        print("#  After fitting image is {img}".format(img=image))
        return image

    def postprocess(self):  # do stuff to canvas
        # for polaroid
        # add title to canvas

        draw = ImageDraw.Draw(self.canvas)

        # font = ImageFont.truetype("BangParty.ttf", 200)
        # font = ImageFont.truetype("CaviarDreams.ttf", 70)
        # font = ImageFont.truetype("sans-serif.ttf", 16)
        # font = ImageFont.truetype("DKUncleEdward.otf", 90)
        font = ImageFont.truetype("fonts/PoplarStd.otf", 80)

        # astr = '''The rain in Spain falls mainly on the plains.'''
        wrapd_title = textwrap.wrap(self.title, width=100)

        MAX_W = self.canvas_width
        # MAX_W = self.canvas_width - 2 * (self.horizontal_padding)
        # font = ImageFont.truetype("retro_party.ttf", 70)
        txt_padding = 10
        _, resized_height = self.images_objs[0].size
        real_title_space = int(self.canvas_height - ((2 * self.vertical_padding) + resized_height))

        nline = len(wrapd_title)
        hline = int(real_title_space // nline)
        h_mid = int(real_title_space // 2)
        start_from_mid = (nline // 2) * hline
        start_h = \
            self.canvas_height \
            - (h_mid + start_from_mid + (1 * real_title_space) +
               (nline // 2 * txt_padding))

        current_h = start_h
        for line in wrapd_title:
            w, h = draw.textsize(line, font=font)
            draw.text(((MAX_W - w) / 2, current_h), line, (0, 0, 0), font=font)
            current_h += hline + txt_padding



class FourPicturesCollage(Collage):
    def __init__(self):
        print("FourPicturesCollage")
        super().__init__()
        print(vars(self))
    # def resize(self):
    #     pass

    def compute_canvas_margins(self):
        # margin canvas
        self.margin_top = 0.01 * self.canvas_height  # 1%
        self.margin_right = 0.01 * self.canvas_width  # 1%
        self.margin_bottom = 0.01 * self.canvas_height  # 1%
        self.margin_left = 0.01 * self.canvas_width  # 1%

    def resize_images(self):
        print("resize for FourPicturesCollage")
        self.images_objs = [
            self.thumbnailise(image, max) for image in self.images_objs
        ]
        print(self.images_objs)
        # max_img_width = int((int(self.canvas_width // self.ncols) - self.x_wsp))
        # max_img_height = int((int(self.canvas_height // self.rows) - self.y_wsp))

        # # max_width = int((int(max_width // self.ncols) - x_wspace))
        # # max_height = int((int(max_height // self.nrows) - y_wspace))
        # # if 4K take the middle
        # # else
        # print("max_width :: ", max_width)
        # print("max_height :: ", max_height)
        # # max_canvas_side = max(max_width, max_height)
        # rsize = (max_width, max_height)
        # # ok
        # narrowest = min(max_img_width, max_img_height)
        # if max_width < max_height:
        #     rsize = (max_height + 1, max_height)
        # image.thumbnail(rsize)
        # return image

        # double check if this works, because modification in place


class GifCollage(FourPicturesCollage):
    def __init__(self):
        super().__init__()


class PopCollage(FourPicturesCollage):
    def __init__(self):
        print("PopCollage")
        super().__init__()
        # this is landscape so width == width
        self.canvas_width = CANVAS_DIMENSIONS[0]
        self.canvas_height = CANVAS_DIMENSIONS[1]
        self.compute_canvas_margins()
        self.canvas = Image.new('RGB',
                                (self.canvas_width, self.canvas_height),
                                (255, 255, 255))
        # pola specific
        self.ncols = 2
        self.nrows = 2
        self.pic_number = self.ncols * self.nrows
        self.pic_shot = 1

        # Some space at bottom for title
        # approximately 15% of the canvas height
        # self.margin_bottom = \
        #     int(0.15 * self.canvas_height) or self.margin_bottom
        # default padding values are enough
        # compute stuff
        self.compute_max_whitespace()
        self.compute_max_image_size()
        # self.title = chose_title(titles=RANDOM_TITLES)
        print(vars(self))

    # def pop_preprocess(image):
    def preprocess(self):
        """
        Replace images objects list with a new list where pictures
        have been processed to add a pop effect
        The processing goes as follow:
        1/ posterizing the 1st (and only) image of the images list
        2/ make a copy of the posterised image
        3/ converting pixel values to HSV color space
        4/ modifying the Hue values of each pixel
        5/ converting back to RGB color space
        6/ adding the processed image to the list
        7/ overwrite the original image list (self.images_objs)
        """
        super().preprocess()
        pop_processed_imgs = []
        image = self.images_objs[0]
        image = ImageOps.posterize(image, 4)
        # get pixel data, as a list
        img_data = list(image.getdata())
        for hue in range(0, 4):
            new_img = image.copy()
            for i in range(0, len(img_data)):
                (h, l, s) = rgb2hls(img_data[i])
                h = hue * 0.25
                img_data[i] = hls2rgb((h, l, s))
            new_img.putdata(img_data)
            pop_processed_imgs.append(new_img)
        self.images_objs = pop_processed_imgs


if __name__ == '__main__':
    print(
    """
         _
     _|#|_
    | (O) |
     -----
    """)

    # canvas_wh = (CSIZE_4IMG_L["width"], CSIZE_4IMG_L["height"])
    # default_border = 100
    # default_gap = default_border // 2
    # resize = resize_normal

    more_pics = True
    while more_pics:
        more_pics = query_yes_no("Take more pictures ?", default="yes")
        if more_pics is True:
            booth_type = query_choices("What type of picture you want :D ?",
                                       BOOTH_TYPES)
            print("debug: booth_type :: " + str(booth_type))

            if not booth_type: booth_type = default_booth

            print("debug: booth_type :: " + str(booth_type))
            if booth_type == 'pola' or booth_type == '':
                collage = PolaroidCollage()
            elif booth_type == 'pop':
                collage = PopCollage()
            elif booth_type == 'gif':
                collage = GifCollage()

            # collage = Collage()
            pics_path = take_pics(number=collage.pic_shot, name_sufix="pic")
            collage.images_objs = [Image.open(pic) for pic in pics_path]

            name = collage.make_collage()
            # subprocess.call(["open", "-a", "/Applications/Preview.app", name])
            tumblr_utils.post_picture(name)

