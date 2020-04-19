cd src
::python demo.py mot --load_model ../models/all_dla34.pth --conf_thres 0.4
::python demo.py mot --load_model ../exp/mot/all_dla34_2/model_5.pth --conf_thres 0.0
python demo_det.py det --load_model ../exp/det/det_dla34/model_last.pth --conf_thres 0.0 --debug 2 --demo ../videos/MOT16-06.mp4
::python demo_det.py det --load_model ../models/all_dla34.pth --debug 2
cd ../