import matplotlib.pyplot as plt

Error = {
         "MLP":{
               "IO_time_L_time":{
                  "tMAE":0.816441957149389,
                  "tMSE":1.018487366164543,
                  "fMAE":1.479827220174263,
                  'fMSE':2.97657377994325},
               "IO_time_L_time+power":{
                  "tMAE":0.816441957149389,
                  "tMSE":1.018487366164543,
                  "fMAE":1.479827220174263,
                  'fMSE':2.97657377994325},
               "IO_time_L_time+PSD":{
                  "tMAE":0.80395322966350,
                  "tMSE":0.97145697976498,
                  "fMAE":1.061274498078280,
                  'fMSE':1.815789195719049},
               "IO_time+PSD":{
                  "tMAE":8.72414420200852,
                  "tMSE":152.7070034570012,
                  "fMAE":4.997815984001426,
                  'fMSE':27.2079834652515},
               "IO_MRA_1D":{
                  "tMAE":0.13908030747297,
                  "tMSE":0.03665468635535,
                  "fMAE":1.122781280748457,
                  'fMSE':1.988398810987143},
               "IO_MRA_2D":{
                  "tMAE":0,
                  "tMSE":0,
                  "fMAE":0,
                  'fMSE':0}},
         "CNN":{
               "IO_time_L_time":{
                  "tMAE":0.061477550190661,
                  "tMSE":0.00709724560006,
                  "fMAE":0.809171134973630,
                  'fMSE':1.180612319934598},
               "IO_time_L_time+power":{
                  "tMAE":0.083735901345411,
                  "tMSE":0.011767935294340,
                  "fMAE":0.86058450699625,
                  'fMSE':1.32270089903027},
               "IO_time_L_time+PSD":{
                  "tMAE":0.075244687311753,
                  "tMSE":0.010048625731506,
                  "fMAE":0.80318187778441,
                  'fMSE':1.191142345406458},
               "IO_time+PSD":{
                  "tMAE":9.27922534171448,
                  "tMSE":92.6740729805024,
                  "fMAE":4.013081524857813,
                  'fMSE':18.2572763927307},
               "IO_MRA_1D":{
                  "tMAE":0.026134565268180,
                  "tMSE":0.001148873583038,
                  "fMAE":0.459751115272310,
                  'fMSE':0.48802275753077},
               "IO_MRA_2D":{
                  "tMAE":0.09020258958503,
                  "tMSE":0.012971127530328,
                  "fMAE":0.70334230970977,
                  'fMSE':0.95064251603460}},
         "LSTM":{
               "IO_time_L_time":{
                  "tMAE":0.133376418698554,
                  "tMSE":0.031312257751998,
                  "fMAE":0.90334329749768,
                  'fMSE':1.413776272992492},
               "IO_time_L_time+power":{
                  "tMAE":0.144867190100688,
                  "tMSE":0.0341115130137945,
                  "fMAE":0.66747748216293,
                  'fMSE':0.88329359527158},
               "IO_time_L_time+PSD":{
                  "tMAE":0.157219259130267,
                  "tMSE":0.03862809270731,
                  "fMAE":1.02264037975052,
                  'fMSE':1.72496240265058},
               "IO_time+PSD":{
                  "tMAE":7.390791465691872,
                  "tMSE":55.61945047325156,
                  "fMAE":0.950461824917455,
                  'fMSE':1.749310827695725},
               "IO_MRA_1D":{
                  "tMAE":0.08785176090379,
                  "tMSE":0.01304035668446,
                  "fMAE":0.66577651452337,
                  'fMSE':.87192048834573},
               "IO_MRA_2D":{
                  "tMAE":0,
                  "tMSE":0,
                  "fMAE":0,
                  'fMSE':0}}}

tMAE = {}
tMSE = {}
fMAE = {}
fMSE = {}

for DNN, dnn in Error.items():
   for Model, model in dnn.items():    
      tMAE.update({DNN+"_"+Model:model["tMAE"]})
      tMSE.update({DNN+"_"+Model:model["tMSE"]})
      fMAE.update({DNN+"_"+Model:model["fMAE"]})
      fMSE.update({DNN+"_"+Model:model["fMSE"]})

def sort_dict_by_value(d, ascending=True):
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=not ascending))
 
sorted_tMAE = sort_dict_by_value(tMAE)
sorted_tMSE = sort_dict_by_value(tMSE)
sorted_fMAE = sort_dict_by_value(fMAE)
sorted_fMSE = sort_dict_by_value(fMSE)

for key, value in sorted_tMAE.items():
   print(key, '\t', value)

# plt.subplot(4, 1, 1)
# plt.bar(list(tMAE.keys()), list(tMAE.values()))
# plt.subplot(4, 1, 2)
# plt.bar(list(tMSE.keys()), list(tMSE.values()))
# plt.subplot(4, 1, 3)
# plt.bar(list(fMAE.keys()), list(fMAE.values()))
# plt.subplot(4, 1, 4)
# plt.bar(list(fMSE.keys()), list(fMSE.values()))
# plt.tight_layout()
# plt.show()