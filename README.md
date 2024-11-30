Code Demo For DeepNM

Requirements:
torch                        1.10.2
torch-cluster                1.6.0
torch-geometric              2.0.4
torch-scatter                2.0.9
torch-sparse                 0.6.13
torch-spline-conv            1.2.1
ot

For reproductivity:

Nodes should be re-id so that initial anchors are labelled as 1-train_size, rest of the test anchors are labelled as train_size+1-al_len

Run the code:

To reproduce DeepNM in Table 3:
python main_douban.py
python main_fbtt.py

Results:
douban.out
fbtt.out

Technical appendix:
Technical_Appendix.pdf
