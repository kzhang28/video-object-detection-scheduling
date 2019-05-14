import matplotlib.pyplot as plt
import os
import re
from pathlib import Path
import argparse
import importlib

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-acc_speed',nargs=1,metavar=('ConfigFile w/o module name'))
    parser.add_argument('-acc_per_snippets',nargs=1,metavar=('ConfigFile w/o module name'))
    args = parser.parse_args()
    # dynamic import
    if  args.acc_speed:
        config_file_wo_extension = args.acc_speed[0].replace('.py', '')
        config = importlib.import_module('plot_config.'+config_file_wo_extension)
    elif args.acc_per_snippets:
        config_file_wo_extension = args.acc_per_snippets[0].replace('.py', '')
        config = importlib.import_module('plot_config.'+config_file_wo_extension)
    else:
        config=None

class Plot:
    def __init__(self,
                 video_list,
                 model_list,
                 evaluation_result_root_dir,
                 purpose_name,
                 res_acc_file_name,
                 res_speed_file_name,
                 speed_dir_name,
                 title_note,
                 res_save_root_dir):

        self.video_list = video_list
        self.model_list = model_list
        self.evaluation_res_root_dir=evaluation_result_root_dir
        self.eval_purpose = purpose_name
        self.speed_dir =speed_dir_name
        self.title_note=title_note
        # Note: this must be specified carefully
        self.acc_file_name=res_acc_file_name # either per_snippets or per_video
        self.speed_file_name=res_speed_file_name
        self.res_save_dir=res_save_root_dir
    def write_meta_data_log(self):
        config_file = config_file_wo_extension+'.py'
        config_file_path = os.path.join(os.getcwd(),'plot_config',config_file)
        save_path =os.path.join(self.res_save_dir,self.eval_purpose,
                                    (self.title_note or 'default_dir'))
        with open(config_file_path) as fd:
            config_content = fd.read()
            with open(os.path.join(save_path,'config_meta.py'),'w') as f:
                f.write(config_content)

    def _plot_speed_acc(self,video_id, lst_acc_per_vd, lst_speed_per_vd,
                        save_dir,title_note=''):
        # plot acc and speed 2d scatter for all models on a single video
        plt.figure()
        for idx, i in enumerate(lst_acc_per_vd):
            plt.scatter(lst_speed_per_vd[idx], lst_acc_per_vd[idx],
                        s=60,label='Model_{}'.format(idx+1))
        plt.title('{}_{}'.format(title_note,video_id))
        plt.xlabel('Processing Time Per Frame')
        plt.ylabel('Video Average Accuracy')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,video_id.replace('/','-')))
    def draw_all_speed_accuracy_2d(self):
        # draw speed accuracy for all models and videos
        for vd in self.video_list:
            pattern = re.compile(vd) # search vd name
            acc_list=[]
            speed_list=[]
            # populate acc and speed list
            for model in self.model_list:
                res_acc_file = os.path.join(self.evaluation_res_root_dir,
                                        self.eval_purpose,model,
                                        self.acc_file_name)
                with open(res_acc_file) as fd:
                    acc = None
                    for line in fd:
                        line=line.strip()
                        match=pattern.search(line)
                        if not match:
                            continue
                        else:
                            line =line.split()
                            acc = float(line[2])
                            break
                    if acc is None:
                        print("[Error]: Fail to find video {}".format(vd))
                        exit()
                    acc_list.append(acc)
                res_speed_file= os.path.join(self.evaluation_res_root_dir,
                                             self.eval_purpose,self.speed_dir,
                                             model,self.speed_file_name)
                with open(res_speed_file) as fd:
                    speed=None
                    for line in fd:
                        line=line.strip()
                        match=pattern.search(line)
                        if not match:
                            continue
                        else:
                            line=line.split()
                            speed=float(line[3][1:-1])
                            break
                    if speed is None:
                        print("[Error]: Fail to find video {}".format(vd))
                        exit()
                    speed_list.append(speed)
            save_dir = os.path.join(self.res_save_dir,self.eval_purpose,
                                    (self.title_note or 'default_dir'))
            Path(save_dir).mkdir(parents=True,exist_ok=True)
            self._plot_speed_acc(vd,acc_list,speed_list,save_dir,self.title_note)
        self.write_meta_data_log()
        plt.show()

    def _plot_per_snippet_acc(self,video,lst_per_model_acc_snippets,save_dir,title_note=''):
        plt.figure()
        for idx, i in enumerate(lst_per_model_acc_snippets):
            plt.plot(i, marker='^', linestyle='--', label='Model_{}'.format(idx+1))
        plt.title('{}_{}'.format(title_note,video))
        plt.xlabel('Snippet ID')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,video.replace('/','-')))
    def draw_all_per_snippet_acc(self):
        # draw per snippet accuracy for all models and videos
        for vd in self.video_list:
            pattern = re.compile(vd)  # search vd name
            lst_acc_per_snippets=[]   # [[acc_of_snippets_of_model_i]]
            for model in self.model_list:
                res_snippets_acc_file = os.path.join(self.evaluation_res_root_dir,
                                            self.eval_purpose, model,
                                            self.acc_file_name)
                with open(res_snippets_acc_file) as fd:
                    accs=None
                    for line in fd:
                        line=line.strip()
                        match = pattern.search(line)
                        if not match:
                            continue
                        else:
                            line = line.split()
                            accs = line[2]
                            break
                    assert accs, "[ERROR]: Not able to find video".format(vd)
                    accs = list(map(float,accs.split(',')))
                    lst_acc_per_snippets.append(accs)
            save_dir = os.path.join(self.res_save_dir, self.eval_purpose,
                                    (self.title_note or 'default_dir'))
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            self._plot_per_snippet_acc(vd,lst_acc_per_snippets,save_dir,self.title_note)
        self.write_meta_data_log()
        plt.show()
    @staticmethod
    def draw_per_frame_accuracy_for_single_video(video_name,model_name,accuracy_lst,save_path):
        plt.figure()
        size = len(accuracy_lst)
        plt.scatter(list(range(size)),accuracy_lst)
        plt.title('{}_{}'.format(video_name,model_name))
        plt.xlabel('frame id')
        plt.ylabel('accuracy')
        plt.savefig(save_path)

if __name__=="__main__":
    plot_obj = Plot(video_list=config.video_list,
                    model_list=config.model_list,
                    evaluation_result_root_dir=config.evaluation_res_root,
                    purpose_name=config.purpose_name,
                    res_acc_file_name=config.acc_file_name,
                    res_speed_file_name=config.speed_file_name,
                    speed_dir_name=config.speed_dir_name,
                    title_note=config.title_note,
                    res_save_root_dir=config.res_save_root_dir)
    if args.acc_speed:
        plot_obj.draw_all_speed_accuracy_2d()
    if args.acc_per_snippets:
        plot_obj.draw_all_per_snippet_acc()

