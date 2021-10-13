import telebot
from telebot import types
from colorizers import *
import os
import argparse
import matplotlib.pyplot as plt

bot = telebot.TeleBot("", parse_mode=None)
bot.remove_webhook()

@bot.message_handler(content_types=['document'])
def color(message):
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = 'photos/' + str(message.chat.id) + '.jpg'  # file_info.file_path
    #print(src)
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
    src = 'photos/' + str(message.chat.id) + '.jpg'
    bot.reply_to(message, "Идет обработка")
    colorphoto(src)
    bot.send_photo(message.chat.id, open(src, 'rb'))
    os.remove(src)
    #colorphoto()

def colorphoto(filename):
    # cv2
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', type=str, default=filename)
    parser.add_argument('--use_gpu', default=False, action='store_true', help='whether to use GPU')
    parser.add_argument('-o', '--save_prefix', type=str, default=filename,
                            help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
    opt = parser.parse_args()
    # load colorizers
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    if (opt.use_gpu):
        colorizer_siggraph17.cuda()
            # default size to process images is 256x256
            # grab L channel in both original ("orig") and resized ("rs") resolutions
    img = load_img(opt.img_path)
    (tens_l_orig, tens_l_rs) = preprocess_img(img)
    if (opt.use_gpu):
        tens_l_rs = tens_l_rs.rocm()

            # colorizer outputs 256x256 ab map
            # resize and concatenate to original L channel
            # img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    plt.imsave(filename, out_img_siggraph17)

if __name__ == '__main__':
    bot.polling(none_stop=True)
