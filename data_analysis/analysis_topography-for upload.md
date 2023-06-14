```python
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
```


```python
df = pd.read_csv('pore_df.csv')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Subject</th>
      <th>Pore Number</th>
      <th>Area (mm²)</th>
      <th>Depth (mm)</th>
      <th>Max Depth (mm)</th>
      <th>Exact Volume (mm3)</th>
      <th>Principal Axis Length (mm)</th>
      <th>Secondary Axis Length (mm)</th>
      <th>Depth Along Principal Axis</th>
      <th>Depth Along Secondary Axis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0.1396</td>
      <td>0.057713</td>
      <td>0.135919</td>
      <td>0.008057</td>
      <td>0.395980</td>
      <td>0.382099</td>
      <td>-0.0399998,-0.045,-0.04,-0.0650002,-0.0879999,...</td>
      <td>-0.0259999,-0.031,-0.0419999,-0.061,-0.0799997...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>0.0660</td>
      <td>0.050416</td>
      <td>0.078150</td>
      <td>0.003327</td>
      <td>0.288444</td>
      <td>0.233238</td>
      <td>-0.04,-0.042,-0.047,-0.054,-0.062,-0.067,-0.07...</td>
      <td>-0.037,-0.0429999,-0.0629999,-0.069,-0.069,-0....</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0.2412</td>
      <td>0.054819</td>
      <td>0.154681</td>
      <td>0.013222</td>
      <td>0.580000</td>
      <td>0.495177</td>
      <td>-0.055,-0.053,-0.046,-0.046,-0.0519998,-0.068,...</td>
      <td>-0.036,-0.0370001,-0.0499999,-0.059,-0.0679999...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>0.1516</td>
      <td>0.054240</td>
      <td>0.120095</td>
      <td>0.008223</td>
      <td>0.438634</td>
      <td>0.368782</td>
      <td>-0.033,-0.0380001,-0.043,-0.0530002,-0.063,-0....</td>
      <td>0.00399999,-0.00200002,-0.0230003,-0.0669998,-...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>0.0928</td>
      <td>0.048031</td>
      <td>0.078854</td>
      <td>0.004457</td>
      <td>0.404969</td>
      <td>0.260768</td>
      <td>-0.039,-0.041,-0.0530001,-0.056,-0.058,-0.059,...</td>
      <td>-0.0160002,-0.04,-0.052,-0.064,-0.0660001,-0.0...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>23791</th>
      <td>110</td>
      <td>311</td>
      <td>0.1220</td>
      <td>0.044792</td>
      <td>0.088566</td>
      <td>0.005465</td>
      <td>0.388330</td>
      <td>0.349285</td>
      <td>-0.048,-0.048,-0.0710005,-0.095,-0.096,-0.097,...</td>
      <td>-0.0389998,-0.0449998,-0.0550006,-0.074,-0.074...</td>
    </tr>
    <tr>
      <th>23792</th>
      <td>110</td>
      <td>312</td>
      <td>0.0520</td>
      <td>0.023058</td>
      <td>0.046342</td>
      <td>0.001199</td>
      <td>0.233238</td>
      <td>0.205913</td>
      <td>-0.0429998,-0.0399998,-0.036,-0.033,-0.033,-0....</td>
      <td>-0.043,-0.045,-0.0459999,-0.04,-0.041,-0.03800...</td>
    </tr>
    <tr>
      <th>23793</th>
      <td>110</td>
      <td>313</td>
      <td>0.1928</td>
      <td>0.070325</td>
      <td>0.120821</td>
      <td>0.013559</td>
      <td>0.564624</td>
      <td>0.420476</td>
      <td>-0.0739999,-0.074,-0.0819999,-0.0790001,-0.077...</td>
      <td>-0.0360004,-0.053,-0.062,-0.068,-0.0769999,-0....</td>
    </tr>
    <tr>
      <th>23794</th>
      <td>110</td>
      <td>314</td>
      <td>0.2152</td>
      <td>0.046200</td>
      <td>0.100157</td>
      <td>0.009942</td>
      <td>0.557853</td>
      <td>0.398497</td>
      <td>-0.0329998,-0.0339999,-0.032,-0.0290001,-0.036...</td>
      <td>-0.0460001,-0.048,-0.0520001,-0.0559999,-0.056...</td>
    </tr>
    <tr>
      <th>23795</th>
      <td>110</td>
      <td>315</td>
      <td>0.0260</td>
      <td>0.038269</td>
      <td>0.065100</td>
      <td>0.000995</td>
      <td>0.200000</td>
      <td>0.100000</td>
      <td>-0.0429995,-0.0469996,-0.0499996,-0.0509995,-0...</td>
      <td>-0.0719999,-0.065,-0.0580002,-0.0509995,-0.039...</td>
    </tr>
  </tbody>
</table>
<p>23796 rows × 10 columns</p>
</div>




```python
test = df['Depth Along Principal Axis']
a = 1

for i in test:
    print(a)
    x = np.array(i.split(',')).astype('float64')
    line = np.array([0.02*(i+1) for i in range(x.size)])
    peaks, _ = find_peaks(x)
    prominences = peak_prominences(x, peaks)[0]
    prominences
    contour_heights = x[peaks] - prominences
    plt.plot(line, x)
    plt.plot((peaks+1)*0.02, x[peaks], "x")
    plt.vlines(x=(peaks+1)*0.02, ymin=contour_heights, ymax=x[peaks])
    plt.show()
    a += 1
```

    1
    


    
![png](output_2_1.png)
    


    # of peaks : 0
    2
    


    
![png](output_2_3.png)
    


    # of peaks : 0
    3
    


    
![png](output_2_5.png)
    


    # of peaks : 0
    4
    


    
![png](output_2_7.png)
    


    # of peaks : 0
    5
    


    
![png](output_2_9.png)
    


    # of peaks : 0
    6
    


    
![png](output_2_11.png)
    


    # of peaks : 0
    7
    


    
![png](output_2_13.png)
    


    # of peaks : 0
    8
    


    
![png](output_2_15.png)
    


    # of peaks : 0
    9
    


    
![png](output_2_17.png)
    


    # of peaks : 0
    10
    


    
![png](output_2_19.png)
    


    # of peaks : 0
    11
    


    
![png](output_2_21.png)
    


    # of peaks : 0
    12
    


    
![png](output_2_23.png)
    


    # of peaks : 0
    13
    


    
![png](output_2_25.png)
    


    # of peaks : 0
    14
    


    
![png](output_2_27.png)
    


    # of peaks : 0
    15
    


    
![png](output_2_29.png)
    


    # of peaks : 1
    16
    


    
![png](output_2_31.png)
    


    # of peaks : 0
    17
    


    
![png](output_2_33.png)
    


    # of peaks : 0
    18
    


    
![png](output_2_35.png)
    


    # of peaks : 0
    19
    


    
![png](output_2_37.png)
    


    # of peaks : 0
    20
    


    
![png](output_2_39.png)
    


    # of peaks : 0
    21
    


    
![png](output_2_41.png)
    


    # of peaks : 0
    22
    


    
![png](output_2_43.png)
    


    # of peaks : 0
    23
    


    
![png](output_2_45.png)
    


    # of peaks : 0
    24
    


    
![png](output_2_47.png)
    


    # of peaks : 0
    25
    


    
![png](output_2_49.png)
    


    # of peaks : 0
    26
    


    
![png](output_2_51.png)
    


    # of peaks : 0
    27
    


    
![png](output_2_53.png)
    


    # of peaks : 0
    28
    


    
![png](output_2_55.png)
    


    # of peaks : 0
    29
    


    
![png](output_2_57.png)
    


    # of peaks : 0
    30
    


    
![png](output_2_59.png)
    


    # of peaks : 0
    31
    


    
![png](output_2_61.png)
    


    # of peaks : 0
    32
    


    
![png](output_2_63.png)
    


    # of peaks : 0
    33
    


    
![png](output_2_65.png)
    


    # of peaks : 0
    34
    


    
![png](output_2_67.png)
    


    # of peaks : 0
    35
    


    
![png](output_2_69.png)
    


    # of peaks : 0
    36
    


    
![png](output_2_71.png)
    


    # of peaks : 0
    37
    


    
![png](output_2_73.png)
    


    # of peaks : 0
    38
    


    
![png](output_2_75.png)
    


    # of peaks : 0
    39
    


    
![png](output_2_77.png)
    


    # of peaks : 0
    40
    


    
![png](output_2_79.png)
    


    # of peaks : 0
    41
    


    
![png](output_2_81.png)
    


    # of peaks : 0
    42
    


    
![png](output_2_83.png)
    


    # of peaks : 0
    43
    


    
![png](output_2_85.png)
    


    # of peaks : 0
    44
    


    
![png](output_2_87.png)
    


    # of peaks : 0
    45
    


    
![png](output_2_89.png)
    


    # of peaks : 0
    46
    


    
![png](output_2_91.png)
    


    # of peaks : 0
    47
    


    
![png](output_2_93.png)
    


    # of peaks : 0
    48
    


    
![png](output_2_95.png)
    


    # of peaks : 0
    49
    


    
![png](output_2_97.png)
    


    # of peaks : 0
    50
    


    
![png](output_2_99.png)
    


    # of peaks : 0
    51
    


    
![png](output_2_101.png)
    


    # of peaks : 0
    52
    


    
![png](output_2_103.png)
    


    # of peaks : 0
    53
    


    
![png](output_2_105.png)
    


    # of peaks : 0
    54
    


    
![png](output_2_107.png)
    


    # of peaks : 0
    55
    


    
![png](output_2_109.png)
    


    # of peaks : 0
    56
    


    
![png](output_2_111.png)
    


    # of peaks : 0
    57
    


    
![png](output_2_113.png)
    


    # of peaks : 0
    58
    


    
![png](output_2_115.png)
    


    # of peaks : 0
    59
    


    
![png](output_2_117.png)
    


    # of peaks : 0
    60
    


    
![png](output_2_119.png)
    


    # of peaks : 0
    61
    


    
![png](output_2_121.png)
    


    # of peaks : 0
    62
    


    
![png](output_2_123.png)
    


    # of peaks : 0
    63
    


    
![png](output_2_125.png)
    


    # of peaks : 0
    64
    


    
![png](output_2_127.png)
    


    # of peaks : 0
    65
    


    
![png](output_2_129.png)
    


    # of peaks : 0
    66
    


    
![png](output_2_131.png)
    


    # of peaks : 0
    67
    


    
![png](output_2_133.png)
    


    # of peaks : 0
    68
    


    
![png](output_2_135.png)
    


    # of peaks : 0
    69
    


    
![png](output_2_137.png)
    


    # of peaks : 0
    70
    


    
![png](output_2_139.png)
    


    # of peaks : 0
    71
    


    
![png](output_2_141.png)
    


    # of peaks : 0
    72
    


    
![png](output_2_143.png)
    


    # of peaks : 0
    73
    


    
![png](output_2_145.png)
    


    # of peaks : 0
    74
    


    
![png](output_2_147.png)
    


    # of peaks : 0
    75
    


    
![png](output_2_149.png)
    


    # of peaks : 0
    76
    


    
![png](output_2_151.png)
    


    # of peaks : 0
    77
    


    
![png](output_2_153.png)
    


    # of peaks : 0
    78
    


    
![png](output_2_155.png)
    


    # of peaks : 0
    79
    


    
![png](output_2_157.png)
    


    # of peaks : 0
    80
    


    
![png](output_2_159.png)
    


    # of peaks : 0
    81
    


    
![png](output_2_161.png)
    


    # of peaks : 0
    82
    


    
![png](output_2_163.png)
    


    # of peaks : 0
    83
    


    
![png](output_2_165.png)
    


    # of peaks : 0
    84
    


    
![png](output_2_167.png)
    


    # of peaks : 0
    85
    


    
![png](output_2_169.png)
    


    # of peaks : 0
    86
    


    
![png](output_2_171.png)
    


    # of peaks : 0
    87
    


    
![png](output_2_173.png)
    


    # of peaks : 0
    88
    


    
![png](output_2_175.png)
    


    # of peaks : 0
    89
    


    
![png](output_2_177.png)
    


    # of peaks : 0
    90
    


    
![png](output_2_179.png)
    


    # of peaks : 0
    91
    


    
![png](output_2_181.png)
    


    # of peaks : 0
    92
    


    
![png](output_2_183.png)
    


    # of peaks : 0
    93
    


    
![png](output_2_185.png)
    


    # of peaks : 0
    94
    


    
![png](output_2_187.png)
    


    # of peaks : 0
    95
    


    
![png](output_2_189.png)
    


    # of peaks : 0
    96
    


    
![png](output_2_191.png)
    


    # of peaks : 0
    97
    


    
![png](output_2_193.png)
    


    # of peaks : 0
    98
    


    
![png](output_2_195.png)
    


    # of peaks : 0
    99
    


    
![png](output_2_197.png)
    


    # of peaks : 0
    100
    


    
![png](output_2_199.png)
    


    # of peaks : 0
    101
    


    
![png](output_2_201.png)
    


    # of peaks : 0
    102
    


    
![png](output_2_203.png)
    


    # of peaks : 0
    103
    


    
![png](output_2_205.png)
    


    # of peaks : 0
    104
    


    
![png](output_2_207.png)
    


    # of peaks : 0
    105
    


    
![png](output_2_209.png)
    


    # of peaks : 0
    106
    


    
![png](output_2_211.png)
    


    # of peaks : 0
    107
    


    
![png](output_2_213.png)
    


    # of peaks : 0
    108
    


    
![png](output_2_215.png)
    


    # of peaks : 0
    109
    


    
![png](output_2_217.png)
    


    # of peaks : 0
    110
    


    
![png](output_2_219.png)
    


    # of peaks : 0
    111
    


    
![png](output_2_221.png)
    


    # of peaks : 0
    112
    


    
![png](output_2_223.png)
    


    # of peaks : 0
    113
    


    
![png](output_2_225.png)
    


    # of peaks : 0
    114
    


    
![png](output_2_227.png)
    


    # of peaks : 0
    115
    


    
![png](output_2_229.png)
    


    # of peaks : 0
    116
    


    
![png](output_2_231.png)
    


    # of peaks : 0
    117
    


    
![png](output_2_233.png)
    


    # of peaks : 0
    118
    


    
![png](output_2_235.png)
    


    # of peaks : 0
    119
    


    
![png](output_2_237.png)
    


    # of peaks : 0
    120
    


    
![png](output_2_239.png)
    


    # of peaks : 0
    121
    


    
![png](output_2_241.png)
    


    # of peaks : 0
    122
    


    
![png](output_2_243.png)
    


    # of peaks : 0
    123
    


    
![png](output_2_245.png)
    


    # of peaks : 0
    124
    


    
![png](output_2_247.png)
    


    # of peaks : 0
    125
    


    
![png](output_2_249.png)
    


    # of peaks : 0
    126
    


    
![png](output_2_251.png)
    


    # of peaks : 0
    127
    


    
![png](output_2_253.png)
    


    # of peaks : 0
    128
    


    
![png](output_2_255.png)
    


    # of peaks : 1
    129
    


    
![png](output_2_257.png)
    


    # of peaks : 0
    130
    


    
![png](output_2_259.png)
    


    # of peaks : 0
    131
    


    
![png](output_2_261.png)
    


    # of peaks : 0
    132
    


    
![png](output_2_263.png)
    


    # of peaks : 0
    133
    


    
![png](output_2_265.png)
    


    # of peaks : 0
    134
    


    
![png](output_2_267.png)
    


    # of peaks : 0
    135
    


    
![png](output_2_269.png)
    


    # of peaks : 0
    136
    


    
![png](output_2_271.png)
    


    # of peaks : 0
    137
    


    
![png](output_2_273.png)
    


    # of peaks : 0
    138
    


    
![png](output_2_275.png)
    


    # of peaks : 0
    139
    


    
![png](output_2_277.png)
    


    # of peaks : 0
    140
    


    
![png](output_2_279.png)
    


    # of peaks : 0
    141
    


    
![png](output_2_281.png)
    


    # of peaks : 0
    142
    


    
![png](output_2_283.png)
    


    # of peaks : 0
    143
    


    
![png](output_2_285.png)
    


    # of peaks : 0
    144
    


    
![png](output_2_287.png)
    


    # of peaks : 0
    145
    


    
![png](output_2_289.png)
    


    # of peaks : 0
    146
    


    
![png](output_2_291.png)
    


    # of peaks : 0
    147
    


    
![png](output_2_293.png)
    


    # of peaks : 0
    148
    


    
![png](output_2_295.png)
    


    # of peaks : 0
    149
    


    
![png](output_2_297.png)
    


    # of peaks : 0
    150
    


    
![png](output_2_299.png)
    


    # of peaks : 0
    151
    


    
![png](output_2_301.png)
    


    # of peaks : 0
    152
    


    
![png](output_2_303.png)
    


    # of peaks : 0
    153
    


    
![png](output_2_305.png)
    


    # of peaks : 0
    154
    


    
![png](output_2_307.png)
    


    # of peaks : 0
    155
    


    
![png](output_2_309.png)
    


    # of peaks : 0
    156
    


    
![png](output_2_311.png)
    


    # of peaks : 0
    157
    


    
![png](output_2_313.png)
    


    # of peaks : 0
    158
    


    
![png](output_2_315.png)
    


    # of peaks : 0
    159
    


    
![png](output_2_317.png)
    


    # of peaks : 0
    160
    


    
![png](output_2_319.png)
    


    # of peaks : 0
    161
    


    
![png](output_2_321.png)
    


    # of peaks : 0
    162
    


    
![png](output_2_323.png)
    


    # of peaks : 0
    163
    


    
![png](output_2_325.png)
    


    # of peaks : 0
    164
    


    
![png](output_2_327.png)
    


    # of peaks : 0
    165
    


    
![png](output_2_329.png)
    


    # of peaks : 0
    166
    


    
![png](output_2_331.png)
    


    # of peaks : 0
    167
    


    
![png](output_2_333.png)
    


    # of peaks : 0
    168
    


    
![png](output_2_335.png)
    


    # of peaks : 0
    169
    


    
![png](output_2_337.png)
    


    # of peaks : 0
    170
    


    
![png](output_2_339.png)
    


    # of peaks : 0
    171
    


    
![png](output_2_341.png)
    


    # of peaks : 0
    172
    


    
![png](output_2_343.png)
    


    # of peaks : 0
    173
    


    
![png](output_2_345.png)
    


    # of peaks : 0
    174
    


    
![png](output_2_347.png)
    


    # of peaks : 0
    175
    


    
![png](output_2_349.png)
    


    # of peaks : 0
    176
    


    
![png](output_2_351.png)
    


    # of peaks : 0
    177
    


    
![png](output_2_353.png)
    


    # of peaks : 1
    178
    


    
![png](output_2_355.png)
    


    # of peaks : 0
    179
    


    
![png](output_2_357.png)
    


    # of peaks : 0
    180
    


    
![png](output_2_359.png)
    


    # of peaks : 1
    181
    


    
![png](output_2_361.png)
    


    # of peaks : 0
    182
    


    
![png](output_2_363.png)
    


    # of peaks : 0
    183
    


    
![png](output_2_365.png)
    


    # of peaks : 0
    184
    


    
![png](output_2_367.png)
    


    # of peaks : 0
    185
    


    
![png](output_2_369.png)
    


    # of peaks : 0
    186
    


    
![png](output_2_371.png)
    


    # of peaks : 0
    187
    


    
![png](output_2_373.png)
    


    # of peaks : 0
    188
    


    
![png](output_2_375.png)
    


    # of peaks : 0
    189
    


    
![png](output_2_377.png)
    


    # of peaks : 0
    190
    


    
![png](output_2_379.png)
    


    # of peaks : 0
    191
    


    
![png](output_2_381.png)
    


    # of peaks : 0
    192
    


    
![png](output_2_383.png)
    


    # of peaks : 0
    193
    


    
![png](output_2_385.png)
    


    # of peaks : 0
    194
    


    
![png](output_2_387.png)
    


    # of peaks : 0
    195
    


    
![png](output_2_389.png)
    


    # of peaks : 0
    196
    


    
![png](output_2_391.png)
    


    # of peaks : 0
    197
    


    
![png](output_2_393.png)
    


    # of peaks : 0
    198
    


    
![png](output_2_395.png)
    


    # of peaks : 0
    199
    


    
![png](output_2_397.png)
    


    # of peaks : 0
    200
    


    
![png](output_2_399.png)
    


    # of peaks : 0
    201
    


    
![png](output_2_401.png)
    


    # of peaks : 0
    202
    


    
![png](output_2_403.png)
    


    # of peaks : 0
    203
    


    
![png](output_2_405.png)
    


    # of peaks : 0
    204
    


    
![png](output_2_407.png)
    


    # of peaks : 0
    205
    


    
![png](output_2_409.png)
    


    # of peaks : 0
    206
    


    
![png](output_2_411.png)
    


    # of peaks : 0
    207
    


    
![png](output_2_413.png)
    


    # of peaks : 0
    208
    


    
![png](output_2_415.png)
    


    # of peaks : 0
    209
    


    
![png](output_2_417.png)
    


    # of peaks : 0
    210
    


    
![png](output_2_419.png)
    


    # of peaks : 0
    211
    


    
![png](output_2_421.png)
    


    # of peaks : 0
    212
    


    
![png](output_2_423.png)
    


    # of peaks : 0
    213
    


    
![png](output_2_425.png)
    


    # of peaks : 0
    214
    


    
![png](output_2_427.png)
    


    # of peaks : 0
    215
    


    
![png](output_2_429.png)
    


    # of peaks : 0
    216
    


    
![png](output_2_431.png)
    


    # of peaks : 0
    217
    


    
![png](output_2_433.png)
    


    # of peaks : 0
    218
    


    
![png](output_2_435.png)
    


    # of peaks : 0
    219
    


    
![png](output_2_437.png)
    


    # of peaks : 0
    220
    


    
![png](output_2_439.png)
    


    # of peaks : 0
    221
    


    
![png](output_2_441.png)
    


    # of peaks : 0
    222
    


    
![png](output_2_443.png)
    


    # of peaks : 0
    223
    


    
![png](output_2_445.png)
    


    # of peaks : 0
    224
    


    
![png](output_2_447.png)
    


    # of peaks : 0
    225
    


    
![png](output_2_449.png)
    


    # of peaks : 0
    226
    


    
![png](output_2_451.png)
    


    # of peaks : 0
    227
    


    
![png](output_2_453.png)
    


    # of peaks : 0
    228
    


    
![png](output_2_455.png)
    


    # of peaks : 2
    229
    


    
![png](output_2_457.png)
    


    # of peaks : 0
    230
    


    
![png](output_2_459.png)
    


    # of peaks : 0
    231
    


    
![png](output_2_461.png)
    


    # of peaks : 0
    232
    


    
![png](output_2_463.png)
    


    # of peaks : 0
    233
    


    
![png](output_2_465.png)
    


    # of peaks : 0
    234
    


    
![png](output_2_467.png)
    


    # of peaks : 0
    235
    


    
![png](output_2_469.png)
    


    # of peaks : 0
    236
    


    
![png](output_2_471.png)
    


    # of peaks : 0
    237
    


    
![png](output_2_473.png)
    


    # of peaks : 0
    238
    


    
![png](output_2_475.png)
    


    # of peaks : 0
    239
    


    
![png](output_2_477.png)
    


    # of peaks : 0
    240
    


    
![png](output_2_479.png)
    


    # of peaks : 0
    241
    


    
![png](output_2_481.png)
    


    # of peaks : 0
    242
    


    
![png](output_2_483.png)
    


    # of peaks : 0
    243
    


    
![png](output_2_485.png)
    


    # of peaks : 0
    244
    


    
![png](output_2_487.png)
    


    # of peaks : 1
    245
    


    
![png](output_2_489.png)
    


    # of peaks : 0
    246
    


    
![png](output_2_491.png)
    


    # of peaks : 0
    247
    


    
![png](output_2_493.png)
    


    # of peaks : 0
    248
    


    
![png](output_2_495.png)
    


    # of peaks : 0
    249
    


    
![png](output_2_497.png)
    


    # of peaks : 0
    250
    


    
![png](output_2_499.png)
    


    # of peaks : 0
    251
    


    
![png](output_2_501.png)
    


    # of peaks : 0
    252
    


    
![png](output_2_503.png)
    


    # of peaks : 0
    253
    


    
![png](output_2_505.png)
    


    # of peaks : 0
    254
    


    
![png](output_2_507.png)
    


    # of peaks : 0
    255
    


    
![png](output_2_509.png)
    


    # of peaks : 0
    256
    


    
![png](output_2_511.png)
    


    # of peaks : 0
    257
    


    
![png](output_2_513.png)
    


    # of peaks : 0
    258
    


    
![png](output_2_515.png)
    


    # of peaks : 0
    259
    


    
![png](output_2_517.png)
    


    # of peaks : 0
    260
    


    
![png](output_2_519.png)
    


    # of peaks : 0
    261
    


    
![png](output_2_521.png)
    


    # of peaks : 0
    262
    


    
![png](output_2_523.png)
    


    # of peaks : 0
    263
    


    
![png](output_2_525.png)
    


    # of peaks : 0
    264
    


    
![png](output_2_527.png)
    


    # of peaks : 0
    265
    


    
![png](output_2_529.png)
    


    # of peaks : 0
    266
    


    
![png](output_2_531.png)
    


    # of peaks : 0
    267
    


    
![png](output_2_533.png)
    


    # of peaks : 0
    268
    


    
![png](output_2_535.png)
    


    # of peaks : 1
    269
    


    
![png](output_2_537.png)
    


    # of peaks : 0
    270
    


    
![png](output_2_539.png)
    


    # of peaks : 0
    271
    


    
![png](output_2_541.png)
    


    # of peaks : 0
    272
    


    
![png](output_2_543.png)
    


    # of peaks : 0
    273
    


    
![png](output_2_545.png)
    


    # of peaks : 0
    274
    


    
![png](output_2_547.png)
    


    # of peaks : 0
    275
    


    
![png](output_2_549.png)
    


    # of peaks : 0
    276
    


    
![png](output_2_551.png)
    


    # of peaks : 0
    277
    


    
![png](output_2_553.png)
    


    # of peaks : 0
    278
    


    
![png](output_2_555.png)
    


    # of peaks : 0
    279
    


    
![png](output_2_557.png)
    


    # of peaks : 0
    280
    


    
![png](output_2_559.png)
    


    # of peaks : 0
    281
    


    
![png](output_2_561.png)
    


    # of peaks : 0
    282
    


    
![png](output_2_563.png)
    


    # of peaks : 0
    283
    


    
![png](output_2_565.png)
    


    # of peaks : 0
    284
    


    
![png](output_2_567.png)
    


    # of peaks : 0
    285
    


    
![png](output_2_569.png)
    


    # of peaks : 0
    286
    


    
![png](output_2_571.png)
    


    # of peaks : 0
    287
    


    
![png](output_2_573.png)
    


    # of peaks : 0
    288
    


    
![png](output_2_575.png)
    


    # of peaks : 0
    289
    


    
![png](output_2_577.png)
    


    # of peaks : 0
    290
    


    
![png](output_2_579.png)
    


    # of peaks : 0
    291
    


    
![png](output_2_581.png)
    


    # of peaks : 0
    292
    


    
![png](output_2_583.png)
    


    # of peaks : 0
    293
    


    
![png](output_2_585.png)
    


    # of peaks : 0
    294
    


    
![png](output_2_587.png)
    


    # of peaks : 0
    295
    


    
![png](output_2_589.png)
    


    # of peaks : 0
    296
    


    
![png](output_2_591.png)
    


    # of peaks : 0
    297
    


    
![png](output_2_593.png)
    


    # of peaks : 0
    298
    


    
![png](output_2_595.png)
    


    # of peaks : 0
    299
    


    
![png](output_2_597.png)
    


    # of peaks : 0
    300
    


    
![png](output_2_599.png)
    


    # of peaks : 0
    301
    


    
![png](output_2_601.png)
    


    # of peaks : 0
    302
    


    
![png](output_2_603.png)
    


    # of peaks : 0
    303
    


    
![png](output_2_605.png)
    


    # of peaks : 0
    304
    


    
![png](output_2_607.png)
    


    # of peaks : 0
    305
    


    
![png](output_2_609.png)
    


    # of peaks : 0
    306
    


    
![png](output_2_611.png)
    


    # of peaks : 0
    307
    


    
![png](output_2_613.png)
    


    # of peaks : 0
    308
    


    
![png](output_2_615.png)
    


    # of peaks : 0
    309
    


    
![png](output_2_617.png)
    


    # of peaks : 0
    310
    


    
![png](output_2_619.png)
    


    # of peaks : 0
    311
    


    
![png](output_2_621.png)
    


    # of peaks : 0
    312
    


    
![png](output_2_623.png)
    


    # of peaks : 0
    313
    


    
![png](output_2_625.png)
    


    # of peaks : 0
    314
    


    
![png](output_2_627.png)
    


    # of peaks : 0
    315
    


    
![png](output_2_629.png)
    


    # of peaks : 0
    316
    


    
![png](output_2_631.png)
    


    # of peaks : 0
    317
    


    
![png](output_2_633.png)
    


    # of peaks : 0
    318
    


    
![png](output_2_635.png)
    


    # of peaks : 0
    319
    


    
![png](output_2_637.png)
    


    # of peaks : 0
    320
    


    
![png](output_2_639.png)
    


    # of peaks : 0
    321
    


    
![png](output_2_641.png)
    


    # of peaks : 0
    322
    


    
![png](output_2_643.png)
    


    # of peaks : 0
    323
    


    
![png](output_2_645.png)
    


    # of peaks : 0
    324
    


    
![png](output_2_647.png)
    


    # of peaks : 0
    325
    


    
![png](output_2_649.png)
    


    # of peaks : 0
    326
    


    
![png](output_2_651.png)
    


    # of peaks : 0
    327
    


    
![png](output_2_653.png)
    


    # of peaks : 0
    328
    


    
![png](output_2_655.png)
    


    # of peaks : 0
    329
    


    
![png](output_2_657.png)
    


    # of peaks : 0
    330
    


    
![png](output_2_659.png)
    


    # of peaks : 0
    331
    


    
![png](output_2_661.png)
    


    # of peaks : 0
    332
    


    
![png](output_2_663.png)
    


    # of peaks : 0
    333
    


    
![png](output_2_665.png)
    


    # of peaks : 0
    334
    


    
![png](output_2_667.png)
    


    # of peaks : 0
    335
    


    
![png](output_2_669.png)
    


    # of peaks : 0
    336
    


    
![png](output_2_671.png)
    


    # of peaks : 0
    337
    


    
![png](output_2_673.png)
    


    # of peaks : 0
    338
    


    
![png](output_2_675.png)
    


    # of peaks : 0
    339
    


    
![png](output_2_677.png)
    


    # of peaks : 0
    340
    


    
![png](output_2_679.png)
    


    # of peaks : 0
    341
    


    
![png](output_2_681.png)
    


    # of peaks : 0
    342
    


    
![png](output_2_683.png)
    


    # of peaks : 0
    343
    


    
![png](output_2_685.png)
    


    # of peaks : 0
    344
    


    
![png](output_2_687.png)
    


    # of peaks : 0
    345
    


    
![png](output_2_689.png)
    


    # of peaks : 0
    346
    


    
![png](output_2_691.png)
    


    # of peaks : 0
    347
    


    
![png](output_2_693.png)
    


    # of peaks : 0
    348
    


    
![png](output_2_695.png)
    


    # of peaks : 0
    349
    


    
![png](output_2_697.png)
    


    # of peaks : 0
    350
    


    
![png](output_2_699.png)
    


    # of peaks : 0
    351
    


    
![png](output_2_701.png)
    


    # of peaks : 0
    352
    


    
![png](output_2_703.png)
    


    # of peaks : 0
    353
    


    
![png](output_2_705.png)
    


    # of peaks : 0
    354
    


    
![png](output_2_707.png)
    


    # of peaks : 0
    355
    


    
![png](output_2_709.png)
    


    # of peaks : 0
    356
    


    
![png](output_2_711.png)
    


    # of peaks : 0
    357
    


    
![png](output_2_713.png)
    


    # of peaks : 0
    358
    


    
![png](output_2_715.png)
    


    # of peaks : 0
    359
    


    
![png](output_2_717.png)
    


    # of peaks : 0
    360
    


    
![png](output_2_719.png)
    


    # of peaks : 0
    361
    


    
![png](output_2_721.png)
    


    # of peaks : 0
    362
    


    
![png](output_2_723.png)
    


    # of peaks : 0
    363
    


    
![png](output_2_725.png)
    


    # of peaks : 0
    364
    


    
![png](output_2_727.png)
    


    # of peaks : 0
    365
    


    
![png](output_2_729.png)
    


    # of peaks : 0
    366
    


    
![png](output_2_731.png)
    


    # of peaks : 0
    367
    


    
![png](output_2_733.png)
    


    # of peaks : 0
    368
    


    
![png](output_2_735.png)
    


    # of peaks : 0
    369
    


    
![png](output_2_737.png)
    


    # of peaks : 0
    370
    


    
![png](output_2_739.png)
    


    # of peaks : 0
    371
    


    
![png](output_2_741.png)
    


    # of peaks : 0
    372
    


    
![png](output_2_743.png)
    


    # of peaks : 0
    373
    


    
![png](output_2_745.png)
    


    # of peaks : 0
    374
    


    
![png](output_2_747.png)
    


    # of peaks : 0
    375
    


    
![png](output_2_749.png)
    


    # of peaks : 0
    376
    


    
![png](output_2_751.png)
    


    # of peaks : 1
    377
    


    
![png](output_2_753.png)
    


    # of peaks : 0
    378
    


    
![png](output_2_755.png)
    


    # of peaks : 0
    379
    


    
![png](output_2_757.png)
    


    # of peaks : 0
    380
    


    
![png](output_2_759.png)
    


    # of peaks : 0
    381
    


    
![png](output_2_761.png)
    


    # of peaks : 0
    382
    


    
![png](output_2_763.png)
    


    # of peaks : 1
    383
    


    
![png](output_2_765.png)
    


    # of peaks : 0
    384
    


    
![png](output_2_767.png)
    


    # of peaks : 0
    385
    


    
![png](output_2_769.png)
    


    # of peaks : 0
    386
    


    
![png](output_2_771.png)
    


    # of peaks : 1
    387
    


    
![png](output_2_773.png)
    


    # of peaks : 0
    388
    


    
![png](output_2_775.png)
    


    # of peaks : 0
    389
    


    
![png](output_2_777.png)
    


    # of peaks : 0
    390
    


    
![png](output_2_779.png)
    


    # of peaks : 0
    391
    


    
![png](output_2_781.png)
    


    # of peaks : 0
    392
    


    
![png](output_2_783.png)
    


    # of peaks : 0
    393
    


    
![png](output_2_785.png)
    


    # of peaks : 0
    394
    


    
![png](output_2_787.png)
    


    # of peaks : 0
    395
    


    
![png](output_2_789.png)
    


    # of peaks : 0
    396
    


    
![png](output_2_791.png)
    


    # of peaks : 0
    397
    


    
![png](output_2_793.png)
    


    # of peaks : 0
    398
    


    
![png](output_2_795.png)
    


    # of peaks : 0
    399
    


    
![png](output_2_797.png)
    


    # of peaks : 0
    400
    


    
![png](output_2_799.png)
    


    # of peaks : 0
    401
    


    
![png](output_2_801.png)
    


    # of peaks : 0
    402
    


    
![png](output_2_803.png)
    


    # of peaks : 0
    403
    


    
![png](output_2_805.png)
    


    # of peaks : 0
    404
    


    
![png](output_2_807.png)
    


    # of peaks : 0
    405
    


    
![png](output_2_809.png)
    


    # of peaks : 0
    406
    


    
![png](output_2_811.png)
    


    # of peaks : 0
    407
    


    
![png](output_2_813.png)
    


    # of peaks : 0
    408
    


    
![png](output_2_815.png)
    


    # of peaks : 0
    409
    


    
![png](output_2_817.png)
    


    # of peaks : 0
    410
    


    
![png](output_2_819.png)
    


    # of peaks : 0
    411
    


    
![png](output_2_821.png)
    


    # of peaks : 0
    412
    


    
![png](output_2_823.png)
    


    # of peaks : 0
    413
    


    
![png](output_2_825.png)
    


    # of peaks : 0
    414
    


    
![png](output_2_827.png)
    


    # of peaks : 0
    415
    


    
![png](output_2_829.png)
    


    # of peaks : 0
    416
    


    
![png](output_2_831.png)
    


    # of peaks : 0
    417
    


    
![png](output_2_833.png)
    


    # of peaks : 0
    418
    


    
![png](output_2_835.png)
    


    # of peaks : 0
    419
    


    
![png](output_2_837.png)
    


    # of peaks : 0
    420
    


    
![png](output_2_839.png)
    


    # of peaks : 0
    421
    


    
![png](output_2_841.png)
    


    # of peaks : 0
    422
    


    
![png](output_2_843.png)
    


    # of peaks : 0
    423
    


    
![png](output_2_845.png)
    


    # of peaks : 0
    424
    


    
![png](output_2_847.png)
    


    # of peaks : 0
    425
    


    
![png](output_2_849.png)
    


    # of peaks : 0
    426
    


    
![png](output_2_851.png)
    


    # of peaks : 0
    427
    


    
![png](output_2_853.png)
    


    # of peaks : 0
    428
    


    
![png](output_2_855.png)
    


    # of peaks : 0
    429
    


    
![png](output_2_857.png)
    


    # of peaks : 0
    430
    


    
![png](output_2_859.png)
    


    # of peaks : 0
    431
    


    
![png](output_2_861.png)
    


    # of peaks : 0
    432
    


    
![png](output_2_863.png)
    


    # of peaks : 0
    433
    


    
![png](output_2_865.png)
    


    # of peaks : 0
    434
    


    
![png](output_2_867.png)
    


    # of peaks : 0
    435
    


    
![png](output_2_869.png)
    


    # of peaks : 0
    436
    


    
![png](output_2_871.png)
    


    # of peaks : 0
    437
    


    
![png](output_2_873.png)
    


    # of peaks : 1
    438
    


    
![png](output_2_875.png)
    


    # of peaks : 0
    439
    


    
![png](output_2_877.png)
    


    # of peaks : 0
    440
    


    
![png](output_2_879.png)
    


    # of peaks : 0
    441
    


    
![png](output_2_881.png)
    


    # of peaks : 0
    442
    


    
![png](output_2_883.png)
    


    # of peaks : 0
    443
    


    
![png](output_2_885.png)
    


    # of peaks : 0
    444
    


    
![png](output_2_887.png)
    


    # of peaks : 0
    445
    


    
![png](output_2_889.png)
    


    # of peaks : 0
    446
    


    
![png](output_2_891.png)
    


    # of peaks : 1
    447
    


    
![png](output_2_893.png)
    


    # of peaks : 0
    448
    


    
![png](output_2_895.png)
    


    # of peaks : 0
    449
    


    
![png](output_2_897.png)
    


    # of peaks : 0
    450
    


    
![png](output_2_899.png)
    


    # of peaks : 0
    451
    


    
![png](output_2_901.png)
    


    # of peaks : 0
    452
    


    
![png](output_2_903.png)
    


    # of peaks : 0
    453
    


    
![png](output_2_905.png)
    


    # of peaks : 0
    454
    


    
![png](output_2_907.png)
    


    # of peaks : 0
    455
    


    
![png](output_2_909.png)
    


    # of peaks : 0
    456
    


    
![png](output_2_911.png)
    


    # of peaks : 0
    457
    


    
![png](output_2_913.png)
    


    # of peaks : 0
    458
    


    
![png](output_2_915.png)
    


    # of peaks : 0
    459
    


    
![png](output_2_917.png)
    


    # of peaks : 0
    460
    


    
![png](output_2_919.png)
    


    # of peaks : 0
    461
    


    
![png](output_2_921.png)
    


    # of peaks : 0
    462
    


    
![png](output_2_923.png)
    


    # of peaks : 0
    463
    


    
![png](output_2_925.png)
    


    # of peaks : 0
    464
    


    
![png](output_2_927.png)
    


    # of peaks : 0
    465
    


    
![png](output_2_929.png)
    


    # of peaks : 0
    466
    


    
![png](output_2_931.png)
    


    # of peaks : 0
    467
    


    
![png](output_2_933.png)
    


    # of peaks : 0
    468
    


    
![png](output_2_935.png)
    


    # of peaks : 0
    469
    


    
![png](output_2_937.png)
    


    # of peaks : 0
    470
    


    
![png](output_2_939.png)
    


    # of peaks : 0
    471
    


    
![png](output_2_941.png)
    


    # of peaks : 0
    472
    


    
![png](output_2_943.png)
    


    # of peaks : 1
    473
    


    
![png](output_2_945.png)
    


    # of peaks : 0
    474
    


    
![png](output_2_947.png)
    


    # of peaks : 0
    475
    


    
![png](output_2_949.png)
    


    # of peaks : 0
    476
    


    
![png](output_2_951.png)
    


    # of peaks : 0
    477
    


    
![png](output_2_953.png)
    


    # of peaks : 0
    478
    


    
![png](output_2_955.png)
    


    # of peaks : 0
    479
    


    
![png](output_2_957.png)
    


    # of peaks : 0
    480
    


    
![png](output_2_959.png)
    


    # of peaks : 0
    481
    


    
![png](output_2_961.png)
    


    # of peaks : 0
    482
    


    
![png](output_2_963.png)
    


    # of peaks : 0
    483
    


    
![png](output_2_965.png)
    


    # of peaks : 0
    484
    


    
![png](output_2_967.png)
    


    # of peaks : 0
    485
    


    
![png](output_2_969.png)
    


    # of peaks : 0
    486
    


    
![png](output_2_971.png)
    


    # of peaks : 1
    487
    


    
![png](output_2_973.png)
    


    # of peaks : 0
    488
    


    
![png](output_2_975.png)
    


    # of peaks : 0
    489
    


    
![png](output_2_977.png)
    


    # of peaks : 0
    490
    


    
![png](output_2_979.png)
    


    # of peaks : 0
    491
    


    
![png](output_2_981.png)
    


    # of peaks : 0
    492
    


    
![png](output_2_983.png)
    


    # of peaks : 0
    493
    


    
![png](output_2_985.png)
    


    # of peaks : 0
    494
    


    
![png](output_2_987.png)
    


    # of peaks : 0
    495
    


    
![png](output_2_989.png)
    


    # of peaks : 0
    496
    


    
![png](output_2_991.png)
    


    # of peaks : 0
    497
    


    
![png](output_2_993.png)
    


    # of peaks : 0
    498
    


    
![png](output_2_995.png)
    


    # of peaks : 0
    499
    


    
![png](output_2_997.png)
    


    # of peaks : 0
    500
    


    
![png](output_2_999.png)
    


    # of peaks : 0
    501
    


    
![png](output_2_1001.png)
    


    # of peaks : 0
    502
    


    
![png](output_2_1003.png)
    


    # of peaks : 0
    503
    


    
![png](output_2_1005.png)
    


    # of peaks : 0
    504
    


    
![png](output_2_1007.png)
    


    # of peaks : 0
    505
    


    
![png](output_2_1009.png)
    


    # of peaks : 0
    506
    


    
![png](output_2_1011.png)
    


    # of peaks : 0
    507
    


    
![png](output_2_1013.png)
    


    # of peaks : 0
    508
    


    
![png](output_2_1015.png)
    


    # of peaks : 0
    509
    


    
![png](output_2_1017.png)
    


    # of peaks : 0
    510
    


    
![png](output_2_1019.png)
    


    # of peaks : 0
    511
    


    
![png](output_2_1021.png)
    


    # of peaks : 0
    512
    


    
![png](output_2_1023.png)
    


    # of peaks : 0
    513
    


    
![png](output_2_1025.png)
    


    # of peaks : 0
    514
    


    
![png](output_2_1027.png)
    


    # of peaks : 0
    515
    


    
![png](output_2_1029.png)
    


    # of peaks : 0
    516
    


    
![png](output_2_1031.png)
    


    # of peaks : 0
    517
    


    
![png](output_2_1033.png)
    


    # of peaks : 0
    518
    


    
![png](output_2_1035.png)
    


    # of peaks : 0
    519
    


    
![png](output_2_1037.png)
    


    # of peaks : 1
    520
    


    
![png](output_2_1039.png)
    


    # of peaks : 0
    521
    


    
![png](output_2_1041.png)
    


    # of peaks : 0
    522
    


    
![png](output_2_1043.png)
    


    # of peaks : 0
    523
    


    
![png](output_2_1045.png)
    


    # of peaks : 0
    524
    


    
![png](output_2_1047.png)
    


    # of peaks : 1
    525
    


    
![png](output_2_1049.png)
    


    # of peaks : 0
    526
    


    
![png](output_2_1051.png)
    


    # of peaks : 0
    527
    


    
![png](output_2_1053.png)
    


    # of peaks : 0
    528
    


    
![png](output_2_1055.png)
    


    # of peaks : 0
    529
    


    
![png](output_2_1057.png)
    


    # of peaks : 0
    530
    


    
![png](output_2_1059.png)
    


    # of peaks : 0
    531
    


    
![png](output_2_1061.png)
    


    # of peaks : 0
    532
    


    
![png](output_2_1063.png)
    


    # of peaks : 0
    533
    


    
![png](output_2_1065.png)
    


    # of peaks : 0
    534
    


    
![png](output_2_1067.png)
    


    # of peaks : 1
    535
    


    
![png](output_2_1069.png)
    


    # of peaks : 0
    536
    


    
![png](output_2_1071.png)
    


    # of peaks : 0
    537
    


    
![png](output_2_1073.png)
    


    # of peaks : 0
    538
    


    
![png](output_2_1075.png)
    


    # of peaks : 0
    539
    


    
![png](output_2_1077.png)
    


    # of peaks : 0
    540
    


    
![png](output_2_1079.png)
    


    # of peaks : 0
    541
    


    
![png](output_2_1081.png)
    


    # of peaks : 1
    542
    


    
![png](output_2_1083.png)
    


    # of peaks : 1
    543
    


    
![png](output_2_1085.png)
    


    # of peaks : 0
    544
    


    
![png](output_2_1087.png)
    


    # of peaks : 0
    545
    


    
![png](output_2_1089.png)
    


    # of peaks : 0
    546
    


    
![png](output_2_1091.png)
    


    # of peaks : 0
    547
    


    
![png](output_2_1093.png)
    


    # of peaks : 1
    548
    


    
![png](output_2_1095.png)
    


    # of peaks : 0
    549
    


    
![png](output_2_1097.png)
    


    # of peaks : 0
    550
    


    
![png](output_2_1099.png)
    


    # of peaks : 0
    551
    


    
![png](output_2_1101.png)
    


    # of peaks : 0
    552
    


    
![png](output_2_1103.png)
    


    # of peaks : 0
    553
    


    
![png](output_2_1105.png)
    


    # of peaks : 0
    554
    


    
![png](output_2_1107.png)
    


    # of peaks : 0
    555
    


    
![png](output_2_1109.png)
    


    # of peaks : 0
    556
    


    
![png](output_2_1111.png)
    


    # of peaks : 0
    557
    


    
![png](output_2_1113.png)
    


    # of peaks : 0
    558
    


    
![png](output_2_1115.png)
    


    # of peaks : 0
    559
    


    
![png](output_2_1117.png)
    


    # of peaks : 0
    560
    


    
![png](output_2_1119.png)
    


    # of peaks : 0
    561
    


    
![png](output_2_1121.png)
    


    # of peaks : 0
    562
    


    
![png](output_2_1123.png)
    


    # of peaks : 0
    563
    


    
![png](output_2_1125.png)
    


    # of peaks : 0
    564
    


    
![png](output_2_1127.png)
    


    # of peaks : 0
    565
    


    
![png](output_2_1129.png)
    


    # of peaks : 0
    566
    


    
![png](output_2_1131.png)
    


    # of peaks : 0
    567
    


    
![png](output_2_1133.png)
    


    # of peaks : 0
    568
    


    
![png](output_2_1135.png)
    


    # of peaks : 0
    569
    


    
![png](output_2_1137.png)
    


    # of peaks : 0
    570
    


    
![png](output_2_1139.png)
    


    # of peaks : 0
    571
    


    
![png](output_2_1141.png)
    


    # of peaks : 0
    572
    


    
![png](output_2_1143.png)
    


    # of peaks : 0
    573
    


    
![png](output_2_1145.png)
    


    # of peaks : 0
    574
    


    
![png](output_2_1147.png)
    


    # of peaks : 0
    575
    


    
![png](output_2_1149.png)
    


    # of peaks : 0
    576
    


    
![png](output_2_1151.png)
    


    # of peaks : 0
    577
    


    
![png](output_2_1153.png)
    


    # of peaks : 1
    578
    


    
![png](output_2_1155.png)
    


    # of peaks : 0
    579
    


    
![png](output_2_1157.png)
    


    # of peaks : 0
    580
    


    
![png](output_2_1159.png)
    


    # of peaks : 0
    581
    


    
![png](output_2_1161.png)
    


    # of peaks : 0
    582
    


    
![png](output_2_1163.png)
    


    # of peaks : 0
    583
    


    
![png](output_2_1165.png)
    


    # of peaks : 0
    584
    


    
![png](output_2_1167.png)
    


    # of peaks : 0
    585
    


    
![png](output_2_1169.png)
    


    # of peaks : 0
    586
    


    
![png](output_2_1171.png)
    


    # of peaks : 0
    587
    


    
![png](output_2_1173.png)
    


    # of peaks : 0
    588
    


    
![png](output_2_1175.png)
    


    # of peaks : 0
    589
    


    
![png](output_2_1177.png)
    


    # of peaks : 0
    590
    


    
![png](output_2_1179.png)
    


    # of peaks : 0
    591
    


    
![png](output_2_1181.png)
    


    # of peaks : 0
    592
    


    
![png](output_2_1183.png)
    


    # of peaks : 0
    593
    


    
![png](output_2_1185.png)
    


    # of peaks : 0
    594
    


    
![png](output_2_1187.png)
    


    # of peaks : 0
    595
    


    
![png](output_2_1189.png)
    


    # of peaks : 0
    596
    


    
![png](output_2_1191.png)
    


    # of peaks : 0
    597
    


    
![png](output_2_1193.png)
    


    # of peaks : 0
    598
    


    
![png](output_2_1195.png)
    


    # of peaks : 0
    599
    


    
![png](output_2_1197.png)
    


    # of peaks : 0
    600
    


    
![png](output_2_1199.png)
    


    # of peaks : 0
    601
    


    
![png](output_2_1201.png)
    


    # of peaks : 0
    602
    


    
![png](output_2_1203.png)
    


    # of peaks : 0
    603
    


    
![png](output_2_1205.png)
    


    # of peaks : 0
    604
    


    
![png](output_2_1207.png)
    


    # of peaks : 0
    605
    


    
![png](output_2_1209.png)
    


    # of peaks : 0
    606
    


    
![png](output_2_1211.png)
    


    # of peaks : 0
    607
    


    
![png](output_2_1213.png)
    


    # of peaks : 0
    608
    


    
![png](output_2_1215.png)
    


    # of peaks : 0
    609
    


    
![png](output_2_1217.png)
    


    # of peaks : 0
    610
    


    
![png](output_2_1219.png)
    


    # of peaks : 1
    611
    


    
![png](output_2_1221.png)
    


    # of peaks : 0
    612
    


    
![png](output_2_1223.png)
    


    # of peaks : 0
    613
    


    
![png](output_2_1225.png)
    


    # of peaks : 0
    614
    


    
![png](output_2_1227.png)
    


    # of peaks : 0
    615
    


    
![png](output_2_1229.png)
    


    # of peaks : 0
    616
    


    
![png](output_2_1231.png)
    


    # of peaks : 0
    617
    


    
![png](output_2_1233.png)
    


    # of peaks : 0
    618
    


    
![png](output_2_1235.png)
    


    # of peaks : 0
    619
    


    
![png](output_2_1237.png)
    


    # of peaks : 0
    620
    


    
![png](output_2_1239.png)
    


    # of peaks : 0
    621
    


    
![png](output_2_1241.png)
    


    # of peaks : 0
    622
    


    
![png](output_2_1243.png)
    


    # of peaks : 0
    623
    


    
![png](output_2_1245.png)
    


    # of peaks : 0
    624
    


    
![png](output_2_1247.png)
    


    # of peaks : 0
    625
    


    
![png](output_2_1249.png)
    


    # of peaks : 0
    626
    


    
![png](output_2_1251.png)
    


    # of peaks : 1
    627
    


    
![png](output_2_1253.png)
    


    # of peaks : 0
    628
    


    
![png](output_2_1255.png)
    


    # of peaks : 0
    629
    


    
![png](output_2_1257.png)
    


    # of peaks : 0
    630
    


    
![png](output_2_1259.png)
    


    # of peaks : 0
    631
    


    
![png](output_2_1261.png)
    


    # of peaks : 0
    632
    


    
![png](output_2_1263.png)
    


    # of peaks : 1
    633
    


    
![png](output_2_1265.png)
    


    # of peaks : 0
    634
    


    
![png](output_2_1267.png)
    


    # of peaks : 0
    635
    


    
![png](output_2_1269.png)
    


    # of peaks : 0
    636
    


    
![png](output_2_1271.png)
    


    # of peaks : 0
    637
    


    
![png](output_2_1273.png)
    


    # of peaks : 0
    638
    


    
![png](output_2_1275.png)
    


    # of peaks : 0
    639
    


    
![png](output_2_1277.png)
    


    # of peaks : 0
    640
    


    
![png](output_2_1279.png)
    


    # of peaks : 1
    641
    


    
![png](output_2_1281.png)
    


    # of peaks : 0
    642
    


    
![png](output_2_1283.png)
    


    # of peaks : 0
    643
    


    
![png](output_2_1285.png)
    


    # of peaks : 0
    644
    


    
![png](output_2_1287.png)
    


    # of peaks : 1
    645
    


    
![png](output_2_1289.png)
    


    # of peaks : 0
    646
    


    
![png](output_2_1291.png)
    


    # of peaks : 0
    647
    


    
![png](output_2_1293.png)
    


    # of peaks : 0
    648
    


    
![png](output_2_1295.png)
    


    # of peaks : 0
    649
    


    
![png](output_2_1297.png)
    


    # of peaks : 0
    650
    


    
![png](output_2_1299.png)
    


    # of peaks : 0
    651
    


    
![png](output_2_1301.png)
    


    # of peaks : 0
    652
    


    
![png](output_2_1303.png)
    


    # of peaks : 0
    653
    


    
![png](output_2_1305.png)
    


    # of peaks : 0
    654
    


    
![png](output_2_1307.png)
    


    # of peaks : 0
    655
    


    
![png](output_2_1309.png)
    


    # of peaks : 0
    656
    


    
![png](output_2_1311.png)
    


    # of peaks : 0
    657
    


    
![png](output_2_1313.png)
    


    # of peaks : 0
    658
    


    
![png](output_2_1315.png)
    


    # of peaks : 0
    659
    


    
![png](output_2_1317.png)
    


    # of peaks : 1
    660
    


    
![png](output_2_1319.png)
    


    # of peaks : 0
    661
    


    
![png](output_2_1321.png)
    


    # of peaks : 0
    662
    


    
![png](output_2_1323.png)
    


    # of peaks : 1
    663
    


    
![png](output_2_1325.png)
    


    # of peaks : 0
    664
    


    
![png](output_2_1327.png)
    


    # of peaks : 2
    665
    


    
![png](output_2_1329.png)
    


    # of peaks : 0
    666
    


    
![png](output_2_1331.png)
    


    # of peaks : 0
    667
    


    
![png](output_2_1333.png)
    


    # of peaks : 0
    668
    


    
![png](output_2_1335.png)
    


    # of peaks : 0
    669
    


    
![png](output_2_1337.png)
    


    # of peaks : 1
    670
    


    
![png](output_2_1339.png)
    


    # of peaks : 0
    671
    


    
![png](output_2_1341.png)
    


    # of peaks : 0
    672
    


    
![png](output_2_1343.png)
    


    # of peaks : 0
    673
    


    
![png](output_2_1345.png)
    


    # of peaks : 0
    674
    


    
![png](output_2_1347.png)
    


    # of peaks : 0
    675
    


    
![png](output_2_1349.png)
    


    # of peaks : 0
    676
    


    
![png](output_2_1351.png)
    


    # of peaks : 0
    677
    


    
![png](output_2_1353.png)
    


    # of peaks : 1
    678
    


    
![png](output_2_1355.png)
    


    # of peaks : 0
    679
    


    
![png](output_2_1357.png)
    


    # of peaks : 0
    680
    


    
![png](output_2_1359.png)
    


    # of peaks : 0
    681
    


    
![png](output_2_1361.png)
    


    # of peaks : 0
    682
    


    
![png](output_2_1363.png)
    


    # of peaks : 0
    683
    


    
![png](output_2_1365.png)
    


    # of peaks : 1
    684
    


    
![png](output_2_1367.png)
    


    # of peaks : 1
    685
    


    
![png](output_2_1369.png)
    


    # of peaks : 0
    686
    


    
![png](output_2_1371.png)
    


    # of peaks : 0
    687
    


    
![png](output_2_1373.png)
    


    # of peaks : 0
    688
    


    
![png](output_2_1375.png)
    


    # of peaks : 0
    689
    


    
![png](output_2_1377.png)
    


    # of peaks : 0
    690
    


    
![png](output_2_1379.png)
    


    # of peaks : 0
    691
    


    
![png](output_2_1381.png)
    


    # of peaks : 0
    692
    


    
![png](output_2_1383.png)
    


    # of peaks : 0
    693
    


    
![png](output_2_1385.png)
    


    # of peaks : 0
    694
    


    
![png](output_2_1387.png)
    


    # of peaks : 0
    695
    


    
![png](output_2_1389.png)
    


    # of peaks : 0
    696
    


    
![png](output_2_1391.png)
    


    # of peaks : 0
    697
    


    
![png](output_2_1393.png)
    


    # of peaks : 1
    698
    


    
![png](output_2_1395.png)
    


    # of peaks : 0
    699
    


    
![png](output_2_1397.png)
    


    # of peaks : 1
    700
    


    
![png](output_2_1399.png)
    


    # of peaks : 0
    701
    


    
![png](output_2_1401.png)
    


    # of peaks : 0
    702
    


    
![png](output_2_1403.png)
    


    # of peaks : 0
    703
    


    
![png](output_2_1405.png)
    


    # of peaks : 0
    704
    


    
![png](output_2_1407.png)
    


    # of peaks : 0
    705
    


    
![png](output_2_1409.png)
    


    # of peaks : 0
    706
    


    
![png](output_2_1411.png)
    


    # of peaks : 0
    707
    


    
![png](output_2_1413.png)
    


    # of peaks : 1
    708
    


    
![png](output_2_1415.png)
    


    # of peaks : 0
    709
    


    
![png](output_2_1417.png)
    


    # of peaks : 0
    710
    


    
![png](output_2_1419.png)
    


    # of peaks : 0
    711
    


    
![png](output_2_1421.png)
    


    # of peaks : 0
    712
    


    
![png](output_2_1423.png)
    


    # of peaks : 0
    713
    


    
![png](output_2_1425.png)
    


    # of peaks : 0
    714
    


    
![png](output_2_1427.png)
    


    # of peaks : 0
    715
    


    
![png](output_2_1429.png)
    


    # of peaks : 0
    716
    


    
![png](output_2_1431.png)
    


    # of peaks : 1
    717
    


    
![png](output_2_1433.png)
    


    # of peaks : 1
    718
    


    
![png](output_2_1435.png)
    


    # of peaks : 0
    719
    


    
![png](output_2_1437.png)
    


    # of peaks : 0
    720
    


    
![png](output_2_1439.png)
    


    # of peaks : 0
    721
    


    
![png](output_2_1441.png)
    


    # of peaks : 1
    722
    


    
![png](output_2_1443.png)
    


    # of peaks : 0
    723
    


    
![png](output_2_1445.png)
    


    # of peaks : 0
    724
    


    
![png](output_2_1447.png)
    


    # of peaks : 0
    725
    


    
![png](output_2_1449.png)
    


    # of peaks : 0
    726
    


    
![png](output_2_1451.png)
    


    # of peaks : 0
    727
    


    
![png](output_2_1453.png)
    


    # of peaks : 0
    728
    


    
![png](output_2_1455.png)
    


    # of peaks : 0
    729
    


    
![png](output_2_1457.png)
    


    # of peaks : 0
    730
    


    
![png](output_2_1459.png)
    


    # of peaks : 0
    731
    


    
![png](output_2_1461.png)
    


    # of peaks : 0
    732
    


    
![png](output_2_1463.png)
    


    # of peaks : 0
    733
    


    
![png](output_2_1465.png)
    


    # of peaks : 0
    734
    


    
![png](output_2_1467.png)
    


    # of peaks : 0
    735
    


    
![png](output_2_1469.png)
    


    # of peaks : 0
    736
    


    
![png](output_2_1471.png)
    


    # of peaks : 0
    737
    


    
![png](output_2_1473.png)
    


    # of peaks : 1
    738
    


    
![png](output_2_1475.png)
    


    # of peaks : 0
    739
    


    
![png](output_2_1477.png)
    


    # of peaks : 0
    740
    


    
![png](output_2_1479.png)
    


    # of peaks : 0
    741
    


    
![png](output_2_1481.png)
    


    # of peaks : 1
    742
    


    
![png](output_2_1483.png)
    


    # of peaks : 0
    743
    


    
![png](output_2_1485.png)
    


    # of peaks : 0
    744
    


    
![png](output_2_1487.png)
    


    # of peaks : 0
    745
    


    
![png](output_2_1489.png)
    


    # of peaks : 0
    746
    


    
![png](output_2_1491.png)
    


    # of peaks : 0
    747
    


    
![png](output_2_1493.png)
    


    # of peaks : 0
    748
    


    
![png](output_2_1495.png)
    


    # of peaks : 0
    749
    


    
![png](output_2_1497.png)
    


    # of peaks : 0
    750
    


    
![png](output_2_1499.png)
    


    # of peaks : 0
    751
    


    
![png](output_2_1501.png)
    


    # of peaks : 0
    752
    


    
![png](output_2_1503.png)
    


    # of peaks : 0
    753
    


    
![png](output_2_1505.png)
    


    # of peaks : 0
    754
    


    
![png](output_2_1507.png)
    


    # of peaks : 0
    755
    


    
![png](output_2_1509.png)
    


    # of peaks : 0
    756
    


    
![png](output_2_1511.png)
    


    # of peaks : 0
    757
    


    
![png](output_2_1513.png)
    


    # of peaks : 0
    758
    


    
![png](output_2_1515.png)
    


    # of peaks : 0
    759
    


    
![png](output_2_1517.png)
    


    # of peaks : 0
    760
    


    
![png](output_2_1519.png)
    


    # of peaks : 0
    761
    


    
![png](output_2_1521.png)
    


    # of peaks : 0
    762
    


    
![png](output_2_1523.png)
    


    # of peaks : 0
    763
    


    
![png](output_2_1525.png)
    


    # of peaks : 0
    764
    


    
![png](output_2_1527.png)
    


    # of peaks : 0
    765
    


    
![png](output_2_1529.png)
    


    # of peaks : 0
    766
    


    
![png](output_2_1531.png)
    


    # of peaks : 0
    767
    


    
![png](output_2_1533.png)
    


    # of peaks : 0
    768
    


    
![png](output_2_1535.png)
    


    # of peaks : 0
    769
    


    
![png](output_2_1537.png)
    


    # of peaks : 0
    770
    


    
![png](output_2_1539.png)
    


    # of peaks : 0
    771
    


    
![png](output_2_1541.png)
    


    # of peaks : 0
    772
    


    
![png](output_2_1543.png)
    


    # of peaks : 0
    773
    


    
![png](output_2_1545.png)
    


    # of peaks : 0
    774
    


    
![png](output_2_1547.png)
    


    # of peaks : 0
    775
    


    
![png](output_2_1549.png)
    


    # of peaks : 0
    776
    


    
![png](output_2_1551.png)
    


    # of peaks : 0
    777
    


    
![png](output_2_1553.png)
    


    # of peaks : 0
    778
    


    
![png](output_2_1555.png)
    


    # of peaks : 0
    779
    


    
![png](output_2_1557.png)
    


    # of peaks : 0
    780
    


    
![png](output_2_1559.png)
    


    # of peaks : 0
    781
    


    
![png](output_2_1561.png)
    


    # of peaks : 0
    782
    


    
![png](output_2_1563.png)
    


    # of peaks : 0
    783
    


    
![png](output_2_1565.png)
    


    # of peaks : 0
    784
    


    
![png](output_2_1567.png)
    


    # of peaks : 0
    785
    


    
![png](output_2_1569.png)
    


    # of peaks : 0
    786
    


    
![png](output_2_1571.png)
    


    # of peaks : 0
    787
    


    
![png](output_2_1573.png)
    


    # of peaks : 0
    788
    


    
![png](output_2_1575.png)
    


    # of peaks : 0
    789
    


    
![png](output_2_1577.png)
    


    # of peaks : 0
    790
    


    
![png](output_2_1579.png)
    


    # of peaks : 0
    791
    


    
![png](output_2_1581.png)
    


    # of peaks : 0
    792
    


    
![png](output_2_1583.png)
    


    # of peaks : 0
    793
    


    
![png](output_2_1585.png)
    


    # of peaks : 0
    794
    


    
![png](output_2_1587.png)
    


    # of peaks : 0
    795
    


    
![png](output_2_1589.png)
    


    # of peaks : 0
    796
    


    
![png](output_2_1591.png)
    


    # of peaks : 0
    797
    


    
![png](output_2_1593.png)
    


    # of peaks : 0
    798
    


    
![png](output_2_1595.png)
    


    # of peaks : 0
    799
    


    
![png](output_2_1597.png)
    


    # of peaks : 0
    800
    


    
![png](output_2_1599.png)
    


    # of peaks : 0
    801
    


    
![png](output_2_1601.png)
    


    # of peaks : 0
    802
    


    
![png](output_2_1603.png)
    


    # of peaks : 0
    803
    


    
![png](output_2_1605.png)
    


    # of peaks : 0
    804
    


    
![png](output_2_1607.png)
    


    # of peaks : 0
    805
    


    
![png](output_2_1609.png)
    


    # of peaks : 0
    806
    


    
![png](output_2_1611.png)
    


    # of peaks : 0
    807
    


    
![png](output_2_1613.png)
    


    # of peaks : 0
    808
    


    
![png](output_2_1615.png)
    


    # of peaks : 0
    809
    


    
![png](output_2_1617.png)
    


    # of peaks : 0
    810
    


    
![png](output_2_1619.png)
    


    # of peaks : 0
    811
    


    
![png](output_2_1621.png)
    


    # of peaks : 0
    812
    


    
![png](output_2_1623.png)
    


    # of peaks : 0
    813
    


    
![png](output_2_1625.png)
    


    # of peaks : 0
    814
    


    
![png](output_2_1627.png)
    


    # of peaks : 0
    815
    


    
![png](output_2_1629.png)
    


    # of peaks : 0
    816
    


    
![png](output_2_1631.png)
    


    # of peaks : 1
    817
    


    
![png](output_2_1633.png)
    


    # of peaks : 0
    818
    


    
![png](output_2_1635.png)
    


    # of peaks : 0
    819
    


    
![png](output_2_1637.png)
    


    # of peaks : 0
    820
    


    
![png](output_2_1639.png)
    


    # of peaks : 0
    821
    


    
![png](output_2_1641.png)
    


    # of peaks : 0
    822
    


    
![png](output_2_1643.png)
    


    # of peaks : 0
    823
    


    
![png](output_2_1645.png)
    


    # of peaks : 0
    824
    


    
![png](output_2_1647.png)
    


    # of peaks : 0
    825
    


    
![png](output_2_1649.png)
    


    # of peaks : 0
    826
    


    
![png](output_2_1651.png)
    


    # of peaks : 0
    827
    


    
![png](output_2_1653.png)
    


    # of peaks : 0
    828
    


    
![png](output_2_1655.png)
    


    # of peaks : 0
    829
    


    
![png](output_2_1657.png)
    


    # of peaks : 0
    830
    


    
![png](output_2_1659.png)
    


    # of peaks : 0
    831
    


    
![png](output_2_1661.png)
    


    # of peaks : 0
    832
    


    
![png](output_2_1663.png)
    


    # of peaks : 0
    833
    


    
![png](output_2_1665.png)
    


    # of peaks : 0
    834
    


    
![png](output_2_1667.png)
    


    # of peaks : 0
    835
    


    
![png](output_2_1669.png)
    


    # of peaks : 0
    836
    


    
![png](output_2_1671.png)
    


    # of peaks : 0
    837
    


    
![png](output_2_1673.png)
    


    # of peaks : 0
    838
    


    
![png](output_2_1675.png)
    


    # of peaks : 0
    839
    


    
![png](output_2_1677.png)
    


    # of peaks : 0
    840
    


    
![png](output_2_1679.png)
    


    # of peaks : 0
    841
    


    
![png](output_2_1681.png)
    


    # of peaks : 0
    842
    


    
![png](output_2_1683.png)
    


    # of peaks : 0
    843
    


    
![png](output_2_1685.png)
    


    # of peaks : 0
    844
    


    
![png](output_2_1687.png)
    


    # of peaks : 0
    845
    


    
![png](output_2_1689.png)
    


    # of peaks : 0
    846
    


    
![png](output_2_1691.png)
    


    # of peaks : 0
    847
    


    
![png](output_2_1693.png)
    


    # of peaks : 0
    848
    


    
![png](output_2_1695.png)
    


    # of peaks : 0
    849
    


    
![png](output_2_1697.png)
    


    # of peaks : 0
    850
    


    
![png](output_2_1699.png)
    


    # of peaks : 0
    851
    


    
![png](output_2_1701.png)
    


    # of peaks : 0
    852
    


    
![png](output_2_1703.png)
    


    # of peaks : 0
    853
    


    
![png](output_2_1705.png)
    


    # of peaks : 0
    854
    


    
![png](output_2_1707.png)
    


    # of peaks : 0
    855
    


    
![png](output_2_1709.png)
    


    # of peaks : 0
    856
    


    
![png](output_2_1711.png)
    


    # of peaks : 0
    857
    


    
![png](output_2_1713.png)
    


    # of peaks : 0
    858
    


    
![png](output_2_1715.png)
    


    # of peaks : 0
    859
    


    
![png](output_2_1717.png)
    


    # of peaks : 0
    860
    


    
![png](output_2_1719.png)
    


    # of peaks : 0
    861
    


    
![png](output_2_1721.png)
    


    # of peaks : 0
    862
    


    
![png](output_2_1723.png)
    


    # of peaks : 0
    863
    


    
![png](output_2_1725.png)
    


    # of peaks : 0
    864
    


    
![png](output_2_1727.png)
    


    # of peaks : 0
    865
    


    
![png](output_2_1729.png)
    


    # of peaks : 0
    866
    


    
![png](output_2_1731.png)
    


    # of peaks : 0
    867
    


    
![png](output_2_1733.png)
    


    # of peaks : 0
    868
    


    
![png](output_2_1735.png)
    


    # of peaks : 0
    869
    


    
![png](output_2_1737.png)
    


    # of peaks : 0
    870
    


    
![png](output_2_1739.png)
    


    # of peaks : 0
    871
    


    
![png](output_2_1741.png)
    


    # of peaks : 0
    872
    


    
![png](output_2_1743.png)
    


    # of peaks : 0
    873
    


    
![png](output_2_1745.png)
    


    # of peaks : 0
    874
    


    
![png](output_2_1747.png)
    


    # of peaks : 0
    875
    


    
![png](output_2_1749.png)
    


    # of peaks : 0
    876
    


    
![png](output_2_1751.png)
    


    # of peaks : 0
    877
    


    
![png](output_2_1753.png)
    


    # of peaks : 0
    878
    


    
![png](output_2_1755.png)
    


    # of peaks : 0
    879
    


    
![png](output_2_1757.png)
    


    # of peaks : 0
    880
    


    
![png](output_2_1759.png)
    


    # of peaks : 0
    881
    


    
![png](output_2_1761.png)
    


    # of peaks : 0
    882
    


    
![png](output_2_1763.png)
    


    # of peaks : 0
    883
    


    
![png](output_2_1765.png)
    


    # of peaks : 0
    884
    


    
![png](output_2_1767.png)
    


    # of peaks : 0
    885
    


    
![png](output_2_1769.png)
    


    # of peaks : 0
    886
    


    
![png](output_2_1771.png)
    


    # of peaks : 0
    887
    


    
![png](output_2_1773.png)
    


    # of peaks : 0
    888
    


    
![png](output_2_1775.png)
    


    # of peaks : 0
    889
    


    
![png](output_2_1777.png)
    


    # of peaks : 0
    890
    


    
![png](output_2_1779.png)
    


    # of peaks : 0
    891
    


    
![png](output_2_1781.png)
    


    # of peaks : 0
    892
    


    
![png](output_2_1783.png)
    


    # of peaks : 0
    893
    


    
![png](output_2_1785.png)
    


    # of peaks : 0
    894
    


    
![png](output_2_1787.png)
    


    # of peaks : 0
    895
    


    
![png](output_2_1789.png)
    


    # of peaks : 0
    896
    


    
![png](output_2_1791.png)
    


    # of peaks : 0
    897
    


    
![png](output_2_1793.png)
    


    # of peaks : 0
    898
    


    
![png](output_2_1795.png)
    


    # of peaks : 0
    899
    


    
![png](output_2_1797.png)
    


    # of peaks : 0
    900
    


    
![png](output_2_1799.png)
    


    # of peaks : 0
    901
    


    
![png](output_2_1801.png)
    


    # of peaks : 0
    902
    


    
![png](output_2_1803.png)
    


    # of peaks : 0
    903
    


    
![png](output_2_1805.png)
    


    # of peaks : 0
    904
    


    
![png](output_2_1807.png)
    


    # of peaks : 0
    905
    


    
![png](output_2_1809.png)
    


    # of peaks : 0
    906
    


    
![png](output_2_1811.png)
    


    # of peaks : 0
    907
    


    
![png](output_2_1813.png)
    


    # of peaks : 0
    908
    


    
![png](output_2_1815.png)
    


    # of peaks : 0
    909
    


    
![png](output_2_1817.png)
    


    # of peaks : 0
    910
    


    
![png](output_2_1819.png)
    


    # of peaks : 0
    911
    


    
![png](output_2_1821.png)
    


    # of peaks : 0
    912
    


    
![png](output_2_1823.png)
    


    # of peaks : 0
    913
    


    
![png](output_2_1825.png)
    


    # of peaks : 0
    914
    


    
![png](output_2_1827.png)
    


    # of peaks : 0
    915
    


    
![png](output_2_1829.png)
    


    # of peaks : 0
    916
    


    
![png](output_2_1831.png)
    


    # of peaks : 0
    917
    


    
![png](output_2_1833.png)
    


    # of peaks : 0
    918
    


    
![png](output_2_1835.png)
    


    # of peaks : 0
    919
    


    
![png](output_2_1837.png)
    


    # of peaks : 0
    920
    


    
![png](output_2_1839.png)
    


    # of peaks : 0
    921
    


    
![png](output_2_1841.png)
    


    # of peaks : 0
    922
    


    
![png](output_2_1843.png)
    


    # of peaks : 0
    923
    


    
![png](output_2_1845.png)
    


    # of peaks : 0
    924
    


    
![png](output_2_1847.png)
    


    # of peaks : 0
    925
    


    
![png](output_2_1849.png)
    


    # of peaks : 0
    926
    


    
![png](output_2_1851.png)
    


    # of peaks : 0
    927
    


    
![png](output_2_1853.png)
    


    # of peaks : 0
    928
    


    
![png](output_2_1855.png)
    


    # of peaks : 0
    929
    


    
![png](output_2_1857.png)
    


    # of peaks : 0
    930
    


    
![png](output_2_1859.png)
    


    # of peaks : 0
    931
    


    
![png](output_2_1861.png)
    


    # of peaks : 0
    932
    


    
![png](output_2_1863.png)
    


    # of peaks : 0
    933
    


    
![png](output_2_1865.png)
    


    # of peaks : 0
    934
    


    
![png](output_2_1867.png)
    


    # of peaks : 0
    935
    


    
![png](output_2_1869.png)
    


    # of peaks : 0
    936
    


    
![png](output_2_1871.png)
    


    # of peaks : 0
    937
    


    
![png](output_2_1873.png)
    


    # of peaks : 0
    938
    


    
![png](output_2_1875.png)
    


    # of peaks : 0
    939
    


    
![png](output_2_1877.png)
    


    # of peaks : 0
    940
    


    
![png](output_2_1879.png)
    


    # of peaks : 0
    941
    


    
![png](output_2_1881.png)
    


    # of peaks : 0
    942
    


    
![png](output_2_1883.png)
    


    # of peaks : 0
    943
    


    
![png](output_2_1885.png)
    


    # of peaks : 0
    944
    


    
![png](output_2_1887.png)
    


    # of peaks : 0
    945
    


    
![png](output_2_1889.png)
    


    # of peaks : 1
    946
    


    
![png](output_2_1891.png)
    


    # of peaks : 0
    947
    


    
![png](output_2_1893.png)
    


    # of peaks : 0
    948
    


    
![png](output_2_1895.png)
    


    # of peaks : 0
    949
    


    
![png](output_2_1897.png)
    


    # of peaks : 0
    950
    


    
![png](output_2_1899.png)
    


    # of peaks : 0
    951
    


    
![png](output_2_1901.png)
    


    # of peaks : 1
    952
    


    
![png](output_2_1903.png)
    


    # of peaks : 0
    953
    


    
![png](output_2_1905.png)
    


    # of peaks : 0
    954
    


    
![png](output_2_1907.png)
    


    # of peaks : 0
    955
    


    
![png](output_2_1909.png)
    


    # of peaks : 0
    956
    


    
![png](output_2_1911.png)
    


    # of peaks : 1
    957
    


    
![png](output_2_1913.png)
    


    # of peaks : 0
    958
    


    
![png](output_2_1915.png)
    


    # of peaks : 0
    959
    


    
![png](output_2_1917.png)
    


    # of peaks : 0
    960
    


    
![png](output_2_1919.png)
    


    # of peaks : 1
    961
    


    
![png](output_2_1921.png)
    


    # of peaks : 1
    962
    


    
![png](output_2_1923.png)
    


    # of peaks : 0
    963
    


    
![png](output_2_1925.png)
    


    # of peaks : 0
    964
    


    
![png](output_2_1927.png)
    


    # of peaks : 0
    965
    


    
![png](output_2_1929.png)
    


    # of peaks : 0
    966
    


    
![png](output_2_1931.png)
    


    # of peaks : 0
    967
    


    
![png](output_2_1933.png)
    


    # of peaks : 0
    968
    


    
![png](output_2_1935.png)
    


    # of peaks : 0
    969
    


    
![png](output_2_1937.png)
    


    # of peaks : 0
    970
    


    
![png](output_2_1939.png)
    


    # of peaks : 0
    971
    


    
![png](output_2_1941.png)
    


    # of peaks : 1
    972
    


    
![png](output_2_1943.png)
    


    # of peaks : 0
    973
    


    
![png](output_2_1945.png)
    


    # of peaks : 0
    974
    


    
![png](output_2_1947.png)
    


    # of peaks : 0
    975
    


    
![png](output_2_1949.png)
    


    # of peaks : 0
    976
    


    
![png](output_2_1951.png)
    


    # of peaks : 1
    977
    


    
![png](output_2_1953.png)
    


    # of peaks : 0
    978
    


    
![png](output_2_1955.png)
    


    # of peaks : 0
    979
    


    
![png](output_2_1957.png)
    


    # of peaks : 0
    980
    


    
![png](output_2_1959.png)
    


    # of peaks : 0
    981
    


    
![png](output_2_1961.png)
    


    # of peaks : 0
    982
    


    
![png](output_2_1963.png)
    


    # of peaks : 0
    983
    


    
![png](output_2_1965.png)
    


    # of peaks : 0
    984
    


    
![png](output_2_1967.png)
    


    # of peaks : 1
    985
    


    
![png](output_2_1969.png)
    


    # of peaks : 0
    986
    


    
![png](output_2_1971.png)
    


    # of peaks : 0
    987
    


    
![png](output_2_1973.png)
    


    # of peaks : 0
    988
    


    
![png](output_2_1975.png)
    


    # of peaks : 0
    989
    


    
![png](output_2_1977.png)
    


    # of peaks : 0
    990
    


    
![png](output_2_1979.png)
    


    # of peaks : 0
    991
    


    
![png](output_2_1981.png)
    


    # of peaks : 0
    992
    


    
![png](output_2_1983.png)
    


    # of peaks : 0
    993
    


    
![png](output_2_1985.png)
    


    # of peaks : 0
    994
    


    
![png](output_2_1987.png)
    


    # of peaks : 1
    995
    


    
![png](output_2_1989.png)
    


    # of peaks : 0
    996
    


    
![png](output_2_1991.png)
    


    # of peaks : 0
    997
    


    
![png](output_2_1993.png)
    


    # of peaks : 1
    998
    


    
![png](output_2_1995.png)
    


    # of peaks : 0
    999
    


    
![png](output_2_1997.png)
    


    # of peaks : 0
    1000
    


    
![png](output_2_1999.png)
    


    # of peaks : 1
    1001
    


    
![png](output_2_2001.png)
    


    # of peaks : 0
    1002
    


    
![png](output_2_2003.png)
    


    # of peaks : 0
    1003
    


    
![png](output_2_2005.png)
    


    # of peaks : 0
    1004
    


    
![png](output_2_2007.png)
    


    # of peaks : 1
    1005
    


    
![png](output_2_2009.png)
    


    # of peaks : 0
    1006
    


    
![png](output_2_2011.png)
    


    # of peaks : 0
    1007
    


    
![png](output_2_2013.png)
    


    # of peaks : 0
    1008
    


    
![png](output_2_2015.png)
    


    # of peaks : 0
    1009
    


    
![png](output_2_2017.png)
    


    # of peaks : 0
    1010
    


    
![png](output_2_2019.png)
    


    # of peaks : 0
    1011
    


    
![png](output_2_2021.png)
    


    # of peaks : 0
    1012
    


    
![png](output_2_2023.png)
    


    # of peaks : 0
    1013
    


    
![png](output_2_2025.png)
    


    # of peaks : 0
    1014
    


    
![png](output_2_2027.png)
    


    # of peaks : 0
    1015
    


    
![png](output_2_2029.png)
    


    # of peaks : 0
    1016
    


    
![png](output_2_2031.png)
    


    # of peaks : 0
    1017
    


    
![png](output_2_2033.png)
    


    # of peaks : 0
    1018
    


    
![png](output_2_2035.png)
    


    # of peaks : 0
    1019
    


    
![png](output_2_2037.png)
    


    # of peaks : 0
    1020
    


    
![png](output_2_2039.png)
    


    # of peaks : 0
    1021
    


    
![png](output_2_2041.png)
    


    # of peaks : 0
    1022
    


    
![png](output_2_2043.png)
    


    # of peaks : 0
    1023
    


    
![png](output_2_2045.png)
    


    # of peaks : 0
    1024
    


    
![png](output_2_2047.png)
    


    # of peaks : 0
    1025
    


    
![png](output_2_2049.png)
    


    # of peaks : 0
    1026
    


    
![png](output_2_2051.png)
    


    # of peaks : 1
    1027
    


    
![png](output_2_2053.png)
    


    # of peaks : 0
    1028
    


    
![png](output_2_2055.png)
    


    # of peaks : 0
    1029
    


    
![png](output_2_2057.png)
    


    # of peaks : 0
    1030
    


    
![png](output_2_2059.png)
    


    # of peaks : 0
    1031
    


    
![png](output_2_2061.png)
    


    # of peaks : 0
    1032
    


    
![png](output_2_2063.png)
    


    # of peaks : 0
    1033
    


    
![png](output_2_2065.png)
    


    # of peaks : 0
    1034
    


    
![png](output_2_2067.png)
    


    # of peaks : 0
    1035
    


    
![png](output_2_2069.png)
    


    # of peaks : 0
    1036
    


    
![png](output_2_2071.png)
    


    # of peaks : 0
    1037
    


    
![png](output_2_2073.png)
    


    # of peaks : 0
    1038
    


    
![png](output_2_2075.png)
    


    # of peaks : 0
    1039
    


    
![png](output_2_2077.png)
    


    # of peaks : 0
    1040
    


    
![png](output_2_2079.png)
    


    # of peaks : 0
    1041
    


    
![png](output_2_2081.png)
    


    # of peaks : 0
    1042
    


    
![png](output_2_2083.png)
    


    # of peaks : 0
    1043
    


    
![png](output_2_2085.png)
    


    # of peaks : 1
    1044
    


    
![png](output_2_2087.png)
    


    # of peaks : 0
    1045
    


    
![png](output_2_2089.png)
    


    # of peaks : 0
    1046
    


    
![png](output_2_2091.png)
    


    # of peaks : 0
    1047
    


    
![png](output_2_2093.png)
    


    # of peaks : 0
    1048
    


    
![png](output_2_2095.png)
    


    # of peaks : 0
    1049
    


    
![png](output_2_2097.png)
    


    # of peaks : 1
    1050
    


    
![png](output_2_2099.png)
    


    # of peaks : 0
    1051
    


    
![png](output_2_2101.png)
    


    # of peaks : 0
    1052
    


    
![png](output_2_2103.png)
    


    # of peaks : 0
    1053
    


    
![png](output_2_2105.png)
    


    # of peaks : 0
    1054
    


    
![png](output_2_2107.png)
    


    # of peaks : 0
    1055
    


    
![png](output_2_2109.png)
    


    # of peaks : 0
    1056
    


    
![png](output_2_2111.png)
    


    # of peaks : 0
    1057
    


    
![png](output_2_2113.png)
    


    # of peaks : 1
    1058
    


    
![png](output_2_2115.png)
    


    # of peaks : 1
    1059
    


    
![png](output_2_2117.png)
    


    # of peaks : 0
    1060
    


    
![png](output_2_2119.png)
    


    # of peaks : 0
    1061
    


    
![png](output_2_2121.png)
    


    # of peaks : 0
    1062
    


    
![png](output_2_2123.png)
    


    # of peaks : 0
    1063
    


    
![png](output_2_2125.png)
    


    # of peaks : 0
    1064
    


    
![png](output_2_2127.png)
    


    # of peaks : 2
    1065
    


    
![png](output_2_2129.png)
    


    # of peaks : 0
    1066
    


    
![png](output_2_2131.png)
    


    # of peaks : 0
    1067
    


    
![png](output_2_2133.png)
    


    # of peaks : 0
    1068
    


    
![png](output_2_2135.png)
    


    # of peaks : 0
    1069
    


    
![png](output_2_2137.png)
    


    # of peaks : 0
    1070
    


    
![png](output_2_2139.png)
    


    # of peaks : 0
    1071
    


    
![png](output_2_2141.png)
    


    # of peaks : 0
    1072
    


    
![png](output_2_2143.png)
    


    # of peaks : 0
    1073
    


    
![png](output_2_2145.png)
    


    # of peaks : 0
    1074
    


    
![png](output_2_2147.png)
    


    # of peaks : 0
    1075
    


    
![png](output_2_2149.png)
    


    # of peaks : 0
    1076
    


    
![png](output_2_2151.png)
    


    # of peaks : 0
    1077
    


    
![png](output_2_2153.png)
    


    # of peaks : 0
    1078
    


    
![png](output_2_2155.png)
    


    # of peaks : 1
    1079
    


    
![png](output_2_2157.png)
    


    # of peaks : 0
    1080
    


    
![png](output_2_2159.png)
    


    # of peaks : 0
    1081
    


    
![png](output_2_2161.png)
    


    # of peaks : 0
    1082
    


    
![png](output_2_2163.png)
    


    # of peaks : 0
    1083
    


    
![png](output_2_2165.png)
    


    # of peaks : 0
    1084
    


    
![png](output_2_2167.png)
    


    # of peaks : 1
    1085
    


    
![png](output_2_2169.png)
    


    # of peaks : 0
    1086
    


    
![png](output_2_2171.png)
    


    # of peaks : 0
    1087
    


    
![png](output_2_2173.png)
    


    # of peaks : 0
    1088
    


    
![png](output_2_2175.png)
    


    # of peaks : 0
    1089
    


    
![png](output_2_2177.png)
    


    # of peaks : 0
    1090
    


    
![png](output_2_2179.png)
    


    # of peaks : 0
    1091
    


    
![png](output_2_2181.png)
    


    # of peaks : 0
    1092
    


    
![png](output_2_2183.png)
    


    # of peaks : 0
    1093
    


    
![png](output_2_2185.png)
    


    # of peaks : 0
    1094
    


    
![png](output_2_2187.png)
    


    # of peaks : 0
    1095
    


    
![png](output_2_2189.png)
    


    # of peaks : 0
    1096
    


    
![png](output_2_2191.png)
    


    # of peaks : 1
    1097
    


    
![png](output_2_2193.png)
    


    # of peaks : 0
    1098
    


    
![png](output_2_2195.png)
    


    # of peaks : 1
    1099
    


    
![png](output_2_2197.png)
    


    # of peaks : 0
    1100
    


    
![png](output_2_2199.png)
    


    # of peaks : 0
    1101
    


    
![png](output_2_2201.png)
    


    # of peaks : 0
    1102
    


    
![png](output_2_2203.png)
    


    # of peaks : 0
    1103
    


    
![png](output_2_2205.png)
    


    # of peaks : 0
    1104
    


    
![png](output_2_2207.png)
    


    # of peaks : 0
    1105
    


    
![png](output_2_2209.png)
    


    # of peaks : 0
    1106
    


    
![png](output_2_2211.png)
    


    # of peaks : 0
    1107
    


    
![png](output_2_2213.png)
    


    # of peaks : 1
    1108
    


    
![png](output_2_2215.png)
    


    # of peaks : 0
    1109
    


    
![png](output_2_2217.png)
    


    # of peaks : 0
    1110
    


    
![png](output_2_2219.png)
    


    # of peaks : 0
    1111
    


    
![png](output_2_2221.png)
    


    # of peaks : 0
    1112
    


    
![png](output_2_2223.png)
    


    # of peaks : 0
    1113
    


    
![png](output_2_2225.png)
    


    # of peaks : 0
    1114
    


    
![png](output_2_2227.png)
    


    # of peaks : 0
    1115
    


    
![png](output_2_2229.png)
    


    # of peaks : 0
    1116
    


    
![png](output_2_2231.png)
    


    # of peaks : 0
    1117
    


    
![png](output_2_2233.png)
    


    # of peaks : 0
    1118
    


    
![png](output_2_2235.png)
    


    # of peaks : 0
    1119
    


    
![png](output_2_2237.png)
    


    # of peaks : 1
    1120
    


    
![png](output_2_2239.png)
    


    # of peaks : 0
    1121
    


    
![png](output_2_2241.png)
    


    # of peaks : 1
    1122
    


    
![png](output_2_2243.png)
    


    # of peaks : 0
    1123
    


    
![png](output_2_2245.png)
    


    # of peaks : 0
    1124
    


    
![png](output_2_2247.png)
    


    # of peaks : 0
    1125
    


    
![png](output_2_2249.png)
    


    # of peaks : 1
    1126
    


    
![png](output_2_2251.png)
    


    # of peaks : 0
    1127
    


    
![png](output_2_2253.png)
    


    # of peaks : 2
    1128
    


    
![png](output_2_2255.png)
    


    # of peaks : 1
    1129
    


    
![png](output_2_2257.png)
    


    # of peaks : 0
    1130
    


    
![png](output_2_2259.png)
    


    # of peaks : 0
    1131
    


    
![png](output_2_2261.png)
    


    # of peaks : 0
    1132
    


    
![png](output_2_2263.png)
    


    # of peaks : 0
    1133
    


    
![png](output_2_2265.png)
    


    # of peaks : 0
    1134
    


    
![png](output_2_2267.png)
    


    # of peaks : 0
    1135
    


    
![png](output_2_2269.png)
    


    # of peaks : 0
    1136
    


    
![png](output_2_2271.png)
    


    # of peaks : 0
    1137
    


    
![png](output_2_2273.png)
    


    # of peaks : 1
    1138
    


    
![png](output_2_2275.png)
    


    # of peaks : 1
    1139
    


    
![png](output_2_2277.png)
    


    # of peaks : 0
    1140
    


    
![png](output_2_2279.png)
    


    # of peaks : 0
    1141
    


    
![png](output_2_2281.png)
    


    # of peaks : 0
    1142
    


    
![png](output_2_2283.png)
    


    # of peaks : 0
    1143
    


    
![png](output_2_2285.png)
    


    # of peaks : 0
    1144
    


    
![png](output_2_2287.png)
    


    # of peaks : 0
    1145
    


    
![png](output_2_2289.png)
    


    # of peaks : 0
    1146
    


    
![png](output_2_2291.png)
    


    # of peaks : 0
    1147
    


    
![png](output_2_2293.png)
    


    # of peaks : 1
    1148
    


    
![png](output_2_2295.png)
    


    # of peaks : 1
    1149
    


    
![png](output_2_2297.png)
    


    # of peaks : 0
    1150
    


    
![png](output_2_2299.png)
    


    # of peaks : 0
    1151
    


    
![png](output_2_2301.png)
    


    # of peaks : 1
    1152
    


    
![png](output_2_2303.png)
    


    # of peaks : 0
    1153
    


    
![png](output_2_2305.png)
    


    # of peaks : 0
    1154
    


    
![png](output_2_2307.png)
    


    # of peaks : 0
    1155
    


    
![png](output_2_2309.png)
    


    # of peaks : 0
    1156
    


    
![png](output_2_2311.png)
    


    # of peaks : 0
    1157
    


    
![png](output_2_2313.png)
    


    # of peaks : 0
    1158
    


    
![png](output_2_2315.png)
    


    # of peaks : 0
    1159
    


    
![png](output_2_2317.png)
    


    # of peaks : 0
    1160
    


    
![png](output_2_2319.png)
    


    # of peaks : 0
    1161
    


    
![png](output_2_2321.png)
    


    # of peaks : 0
    1162
    


    
![png](output_2_2323.png)
    


    # of peaks : 0
    1163
    


    
![png](output_2_2325.png)
    


    # of peaks : 0
    1164
    


    
![png](output_2_2327.png)
    


    # of peaks : 0
    1165
    


    
![png](output_2_2329.png)
    


    # of peaks : 0
    1166
    


    
![png](output_2_2331.png)
    


    # of peaks : 0
    1167
    


    
![png](output_2_2333.png)
    


    # of peaks : 0
    1168
    


    
![png](output_2_2335.png)
    


    # of peaks : 0
    1169
    


    
![png](output_2_2337.png)
    


    # of peaks : 0
    1170
    


    
![png](output_2_2339.png)
    


    # of peaks : 0
    1171
    


    
![png](output_2_2341.png)
    


    # of peaks : 0
    1172
    


    
![png](output_2_2343.png)
    


    # of peaks : 0
    1173
    


    
![png](output_2_2345.png)
    


    # of peaks : 0
    1174
    


    
![png](output_2_2347.png)
    


    # of peaks : 0
    1175
    


    
![png](output_2_2349.png)
    


    # of peaks : 0
    1176
    


    
![png](output_2_2351.png)
    


    # of peaks : 0
    1177
    


    
![png](output_2_2353.png)
    


    # of peaks : 0
    1178
    


    
![png](output_2_2355.png)
    


    # of peaks : 0
    1179
    


    
![png](output_2_2357.png)
    


    # of peaks : 0
    1180
    


    
![png](output_2_2359.png)
    


    # of peaks : 0
    1181
    


    
![png](output_2_2361.png)
    


    # of peaks : 0
    1182
    


    
![png](output_2_2363.png)
    


    # of peaks : 0
    1183
    


    
![png](output_2_2365.png)
    


    # of peaks : 0
    1184
    


    
![png](output_2_2367.png)
    


    # of peaks : 0
    1185
    


    
![png](output_2_2369.png)
    


    # of peaks : 0
    1186
    


    
![png](output_2_2371.png)
    


    # of peaks : 0
    1187
    


    
![png](output_2_2373.png)
    


    # of peaks : 0
    1188
    


    
![png](output_2_2375.png)
    


    # of peaks : 0
    1189
    


    
![png](output_2_2377.png)
    


    # of peaks : 0
    1190
    


    
![png](output_2_2379.png)
    


    # of peaks : 0
    1191
    


    
![png](output_2_2381.png)
    


    # of peaks : 1
    1192
    


    
![png](output_2_2383.png)
    


    # of peaks : 0
    1193
    


    
![png](output_2_2385.png)
    


    # of peaks : 0
    1194
    


    
![png](output_2_2387.png)
    


    # of peaks : 0
    1195
    


    
![png](output_2_2389.png)
    


    # of peaks : 0
    1196
    


    
![png](output_2_2391.png)
    


    # of peaks : 0
    1197
    


    
![png](output_2_2393.png)
    


    # of peaks : 0
    1198
    


    
![png](output_2_2395.png)
    


    # of peaks : 0
    1199
    


    
![png](output_2_2397.png)
    


    # of peaks : 0
    1200
    


    
![png](output_2_2399.png)
    


    # of peaks : 0
    1201
    


    
![png](output_2_2401.png)
    


    # of peaks : 0
    1202
    


    
![png](output_2_2403.png)
    


    # of peaks : 0
    1203
    


    
![png](output_2_2405.png)
    


    # of peaks : 0
    1204
    


    
![png](output_2_2407.png)
    


    # of peaks : 0
    1205
    


    
![png](output_2_2409.png)
    


    # of peaks : 0
    1206
    


    
![png](output_2_2411.png)
    


    # of peaks : 0
    1207
    


    
![png](output_2_2413.png)
    


    # of peaks : 0
    1208
    


    
![png](output_2_2415.png)
    


    # of peaks : 0
    1209
    


    
![png](output_2_2417.png)
    


    # of peaks : 0
    1210
    


    
![png](output_2_2419.png)
    


    # of peaks : 0
    1211
    


    
![png](output_2_2421.png)
    


    # of peaks : 0
    1212
    


    
![png](output_2_2423.png)
    


    # of peaks : 0
    1213
    


    
![png](output_2_2425.png)
    


    # of peaks : 0
    1214
    


    
![png](output_2_2427.png)
    


    # of peaks : 0
    1215
    


    
![png](output_2_2429.png)
    


    # of peaks : 0
    1216
    


    
![png](output_2_2431.png)
    


    # of peaks : 0
    1217
    


    
![png](output_2_2433.png)
    


    # of peaks : 0
    1218
    


    
![png](output_2_2435.png)
    


    # of peaks : 1
    1219
    


    
![png](output_2_2437.png)
    


    # of peaks : 1
    1220
    


    
![png](output_2_2439.png)
    


    # of peaks : 0
    1221
    


    
![png](output_2_2441.png)
    


    # of peaks : 0
    1222
    


    
![png](output_2_2443.png)
    


    # of peaks : 0
    1223
    


    
![png](output_2_2445.png)
    


    # of peaks : 0
    1224
    


    
![png](output_2_2447.png)
    


    # of peaks : 0
    1225
    


    
![png](output_2_2449.png)
    


    # of peaks : 0
    1226
    


    
![png](output_2_2451.png)
    


    # of peaks : 0
    1227
    


    
![png](output_2_2453.png)
    


    # of peaks : 0
    1228
    


    
![png](output_2_2455.png)
    


    # of peaks : 0
    1229
    


    
![png](output_2_2457.png)
    


    # of peaks : 0
    1230
    


    
![png](output_2_2459.png)
    


    # of peaks : 0
    1231
    


    
![png](output_2_2461.png)
    


    # of peaks : 0
    1232
    


    
![png](output_2_2463.png)
    


    # of peaks : 0
    1233
    


    
![png](output_2_2465.png)
    


    # of peaks : 1
    1234
    


    
![png](output_2_2467.png)
    


    # of peaks : 0
    1235
    


    
![png](output_2_2469.png)
    


    # of peaks : 0
    1236
    


    
![png](output_2_2471.png)
    


    # of peaks : 0
    1237
    


    
![png](output_2_2473.png)
    


    # of peaks : 1
    1238
    


    
![png](output_2_2475.png)
    


    # of peaks : 1
    1239
    


    
![png](output_2_2477.png)
    


    # of peaks : 0
    1240
    


    
![png](output_2_2479.png)
    


    # of peaks : 0
    1241
    


    
![png](output_2_2481.png)
    


    # of peaks : 0
    1242
    


    
![png](output_2_2483.png)
    


    # of peaks : 0
    1243
    


    
![png](output_2_2485.png)
    


    # of peaks : 0
    1244
    


    
![png](output_2_2487.png)
    


    # of peaks : 0
    1245
    


    
![png](output_2_2489.png)
    


    # of peaks : 0
    1246
    


    
![png](output_2_2491.png)
    


    # of peaks : 0
    1247
    


    
![png](output_2_2493.png)
    


    # of peaks : 0
    1248
    


    
![png](output_2_2495.png)
    


    # of peaks : 0
    1249
    


    
![png](output_2_2497.png)
    


    # of peaks : 0
    1250
    


    
![png](output_2_2499.png)
    


    # of peaks : 0
    1251
    


    
![png](output_2_2501.png)
    


    # of peaks : 0
    1252
    


    
![png](output_2_2503.png)
    


    # of peaks : 0
    1253
    


    
![png](output_2_2505.png)
    


    # of peaks : 0
    1254
    


    
![png](output_2_2507.png)
    


    # of peaks : 0
    1255
    


    
![png](output_2_2509.png)
    


    # of peaks : 1
    1256
    


    
![png](output_2_2511.png)
    


    # of peaks : 0
    1257
    


    
![png](output_2_2513.png)
    


    # of peaks : 1
    1258
    


    
![png](output_2_2515.png)
    


    # of peaks : 0
    1259
    


    
![png](output_2_2517.png)
    


    # of peaks : 0
    1260
    


    
![png](output_2_2519.png)
    


    # of peaks : 0
    1261
    


    
![png](output_2_2521.png)
    


    # of peaks : 0
    1262
    


    
![png](output_2_2523.png)
    


    # of peaks : 0
    1263
    


    
![png](output_2_2525.png)
    


    # of peaks : 0
    1264
    


    
![png](output_2_2527.png)
    


    # of peaks : 0
    1265
    


    
![png](output_2_2529.png)
    


    # of peaks : 1
    1266
    


    
![png](output_2_2531.png)
    


    # of peaks : 0
    1267
    


    
![png](output_2_2533.png)
    


    # of peaks : 0
    1268
    


    
![png](output_2_2535.png)
    


    # of peaks : 0
    1269
    


    
![png](output_2_2537.png)
    


    # of peaks : 0
    1270
    


    
![png](output_2_2539.png)
    


    # of peaks : 0
    1271
    


    
![png](output_2_2541.png)
    


    # of peaks : 0
    1272
    


    
![png](output_2_2543.png)
    


    # of peaks : 0
    1273
    


    
![png](output_2_2545.png)
    


    # of peaks : 0
    1274
    


    
![png](output_2_2547.png)
    


    # of peaks : 0
    1275
    


    
![png](output_2_2549.png)
    


    # of peaks : 0
    1276
    


    
![png](output_2_2551.png)
    


    # of peaks : 0
    1277
    


    
![png](output_2_2553.png)
    


    # of peaks : 0
    1278
    


    
![png](output_2_2555.png)
    


    # of peaks : 0
    1279
    


    
![png](output_2_2557.png)
    


    # of peaks : 0
    1280
    


    
![png](output_2_2559.png)
    


    # of peaks : 0
    1281
    


    
![png](output_2_2561.png)
    


    # of peaks : 0
    1282
    


    
![png](output_2_2563.png)
    


    # of peaks : 0
    1283
    


    
![png](output_2_2565.png)
    


    # of peaks : 0
    1284
    


    
![png](output_2_2567.png)
    


    # of peaks : 0
    1285
    


    
![png](output_2_2569.png)
    


    # of peaks : 0
    1286
    


    
![png](output_2_2571.png)
    


    # of peaks : 0
    1287
    


    
![png](output_2_2573.png)
    


    # of peaks : 0
    1288
    


    
![png](output_2_2575.png)
    


    # of peaks : 0
    1289
    


    
![png](output_2_2577.png)
    


    # of peaks : 0
    1290
    


    
![png](output_2_2579.png)
    


    # of peaks : 0
    1291
    


    
![png](output_2_2581.png)
    


    # of peaks : 0
    1292
    


    
![png](output_2_2583.png)
    


    # of peaks : 0
    1293
    


    
![png](output_2_2585.png)
    


    # of peaks : 0
    1294
    


    
![png](output_2_2587.png)
    


    # of peaks : 0
    1295
    


    
![png](output_2_2589.png)
    


    # of peaks : 0
    1296
    


    
![png](output_2_2591.png)
    


    # of peaks : 0
    1297
    


    
![png](output_2_2593.png)
    


    # of peaks : 0
    1298
    


    
![png](output_2_2595.png)
    


    # of peaks : 0
    1299
    


    
![png](output_2_2597.png)
    


    # of peaks : 0
    1300
    


    
![png](output_2_2599.png)
    


    # of peaks : 0
    1301
    


    
![png](output_2_2601.png)
    


    # of peaks : 0
    1302
    


    
![png](output_2_2603.png)
    


    # of peaks : 0
    1303
    


    
![png](output_2_2605.png)
    


    # of peaks : 0
    1304
    


    
![png](output_2_2607.png)
    


    # of peaks : 0
    1305
    


    
![png](output_2_2609.png)
    


    # of peaks : 0
    1306
    


    
![png](output_2_2611.png)
    


    # of peaks : 0
    1307
    


    
![png](output_2_2613.png)
    


    # of peaks : 0
    1308
    


    
![png](output_2_2615.png)
    


    # of peaks : 1
    1309
    


    
![png](output_2_2617.png)
    


    # of peaks : 0
    1310
    


    
![png](output_2_2619.png)
    


    # of peaks : 0
    1311
    


    
![png](output_2_2621.png)
    


    # of peaks : 0
    1312
    


    
![png](output_2_2623.png)
    


    # of peaks : 0
    1313
    


    
![png](output_2_2625.png)
    


    # of peaks : 0
    1314
    


    
![png](output_2_2627.png)
    


    # of peaks : 0
    1315
    


    
![png](output_2_2629.png)
    


    # of peaks : 0
    1316
    


    
![png](output_2_2631.png)
    


    # of peaks : 0
    1317
    


    
![png](output_2_2633.png)
    


    # of peaks : 0
    1318
    


    
![png](output_2_2635.png)
    


    # of peaks : 0
    1319
    


    
![png](output_2_2637.png)
    


    # of peaks : 0
    1320
    


    
![png](output_2_2639.png)
    


    # of peaks : 0
    1321
    


    
![png](output_2_2641.png)
    


    # of peaks : 0
    1322
    


    
![png](output_2_2643.png)
    


    # of peaks : 0
    1323
    


    
![png](output_2_2645.png)
    


    # of peaks : 0
    1324
    


    
![png](output_2_2647.png)
    


    # of peaks : 0
    1325
    


    
![png](output_2_2649.png)
    


    # of peaks : 0
    1326
    


    
![png](output_2_2651.png)
    


    # of peaks : 1
    1327
    


    
![png](output_2_2653.png)
    


    # of peaks : 1
    1328
    


    
![png](output_2_2655.png)
    


    # of peaks : 0
    1329
    


    
![png](output_2_2657.png)
    


    # of peaks : 0
    1330
    


    
![png](output_2_2659.png)
    


    # of peaks : 1
    1331
    


    
![png](output_2_2661.png)
    


    # of peaks : 0
    1332
    


    
![png](output_2_2663.png)
    


    # of peaks : 0
    1333
    


    
![png](output_2_2665.png)
    


    # of peaks : 0
    1334
    


    
![png](output_2_2667.png)
    


    # of peaks : 0
    1335
    


    
![png](output_2_2669.png)
    


    # of peaks : 0
    1336
    


    
![png](output_2_2671.png)
    


    # of peaks : 0
    1337
    


    
![png](output_2_2673.png)
    


    # of peaks : 0
    1338
    


    
![png](output_2_2675.png)
    


    # of peaks : 0
    1339
    


    
![png](output_2_2677.png)
    


    # of peaks : 0
    1340
    


    
![png](output_2_2679.png)
    


    # of peaks : 0
    1341
    


    
![png](output_2_2681.png)
    


    # of peaks : 0
    1342
    


    
![png](output_2_2683.png)
    


    # of peaks : 1
    1343
    


    
![png](output_2_2685.png)
    


    # of peaks : 0
    1344
    


    
![png](output_2_2687.png)
    


    # of peaks : 1
    1345
    


    
![png](output_2_2689.png)
    


    # of peaks : 0
    1346
    


    
![png](output_2_2691.png)
    


    # of peaks : 0
    1347
    


    
![png](output_2_2693.png)
    


    # of peaks : 0
    1348
    


    
![png](output_2_2695.png)
    


    # of peaks : 0
    1349
    


    
![png](output_2_2697.png)
    


    # of peaks : 0
    1350
    


    
![png](output_2_2699.png)
    


    # of peaks : 0
    1351
    


    
![png](output_2_2701.png)
    


    # of peaks : 0
    1352
    


    
![png](output_2_2703.png)
    


    # of peaks : 0
    1353
    


    
![png](output_2_2705.png)
    


    # of peaks : 0
    1354
    


    
![png](output_2_2707.png)
    


    # of peaks : 0
    1355
    


    
![png](output_2_2709.png)
    


    # of peaks : 0
    1356
    


    
![png](output_2_2711.png)
    


    # of peaks : 0
    1357
    


    
![png](output_2_2713.png)
    


    # of peaks : 0
    1358
    


    
![png](output_2_2715.png)
    


    # of peaks : 1
    1359
    


    
![png](output_2_2717.png)
    


    # of peaks : 0
    1360
    


    
![png](output_2_2719.png)
    


    # of peaks : 0
    1361
    


    
![png](output_2_2721.png)
    


    # of peaks : 0
    1362
    


    
![png](output_2_2723.png)
    


    # of peaks : 0
    1363
    


    
![png](output_2_2725.png)
    


    # of peaks : 0
    1364
    


    
![png](output_2_2727.png)
    


    # of peaks : 0
    1365
    


    
![png](output_2_2729.png)
    


    # of peaks : 0
    1366
    


    
![png](output_2_2731.png)
    


    # of peaks : 0
    1367
    


    
![png](output_2_2733.png)
    


    # of peaks : 0
    1368
    


    
![png](output_2_2735.png)
    


    # of peaks : 0
    1369
    


    
![png](output_2_2737.png)
    


    # of peaks : 0
    1370
    


    
![png](output_2_2739.png)
    


    # of peaks : 0
    1371
    


    
![png](output_2_2741.png)
    


    # of peaks : 0
    1372
    


    
![png](output_2_2743.png)
    


    # of peaks : 0
    1373
    


    
![png](output_2_2745.png)
    


    # of peaks : 1
    1374
    


    
![png](output_2_2747.png)
    


    # of peaks : 0
    1375
    


    
![png](output_2_2749.png)
    


    # of peaks : 0
    1376
    


    
![png](output_2_2751.png)
    


    # of peaks : 0
    1377
    


    
![png](output_2_2753.png)
    


    # of peaks : 0
    1378
    


    
![png](output_2_2755.png)
    


    # of peaks : 0
    1379
    


    
![png](output_2_2757.png)
    


    # of peaks : 0
    1380
    


    
![png](output_2_2759.png)
    


    # of peaks : 0
    1381
    


    
![png](output_2_2761.png)
    


    # of peaks : 0
    1382
    


    
![png](output_2_2763.png)
    


    # of peaks : 0
    1383
    


    
![png](output_2_2765.png)
    


    # of peaks : 0
    1384
    


    
![png](output_2_2767.png)
    


    # of peaks : 0
    1385
    


    
![png](output_2_2769.png)
    


    # of peaks : 0
    1386
    


    
![png](output_2_2771.png)
    


    # of peaks : 0
    1387
    


    
![png](output_2_2773.png)
    


    # of peaks : 0
    1388
    


    
![png](output_2_2775.png)
    


    # of peaks : 0
    1389
    


    
![png](output_2_2777.png)
    


    # of peaks : 0
    1390
    


    
![png](output_2_2779.png)
    


    # of peaks : 0
    1391
    


    
![png](output_2_2781.png)
    


    # of peaks : 0
    1392
    


    
![png](output_2_2783.png)
    


    # of peaks : 0
    1393
    


    
![png](output_2_2785.png)
    


    # of peaks : 0
    1394
    


    
![png](output_2_2787.png)
    


    # of peaks : 0
    1395
    


    
![png](output_2_2789.png)
    


    # of peaks : 0
    1396
    


    
![png](output_2_2791.png)
    


    # of peaks : 0
    1397
    


    
![png](output_2_2793.png)
    


    # of peaks : 0
    1398
    


    
![png](output_2_2795.png)
    


    # of peaks : 0
    1399
    


    
![png](output_2_2797.png)
    


    # of peaks : 0
    1400
    


    
![png](output_2_2799.png)
    


    # of peaks : 0
    1401
    


    
![png](output_2_2801.png)
    


    # of peaks : 0
    1402
    


    
![png](output_2_2803.png)
    


    # of peaks : 1
    1403
    


    
![png](output_2_2805.png)
    


    # of peaks : 0
    1404
    


    
![png](output_2_2807.png)
    


    # of peaks : 0
    1405
    


    
![png](output_2_2809.png)
    


    # of peaks : 0
    1406
    


    
![png](output_2_2811.png)
    


    # of peaks : 0
    1407
    


    
![png](output_2_2813.png)
    


    # of peaks : 0
    1408
    


    
![png](output_2_2815.png)
    


    # of peaks : 2
    1409
    


    
![png](output_2_2817.png)
    


    # of peaks : 0
    1410
    


    
![png](output_2_2819.png)
    


    # of peaks : 0
    1411
    


    
![png](output_2_2821.png)
    


    # of peaks : 0
    1412
    


    
![png](output_2_2823.png)
    


    # of peaks : 0
    1413
    


    
![png](output_2_2825.png)
    


    # of peaks : 0
    1414
    


    
![png](output_2_2827.png)
    


    # of peaks : 0
    1415
    


    
![png](output_2_2829.png)
    


    # of peaks : 0
    1416
    


    
![png](output_2_2831.png)
    


    # of peaks : 0
    1417
    


    
![png](output_2_2833.png)
    


    # of peaks : 0
    1418
    


    
![png](output_2_2835.png)
    


    # of peaks : 0
    1419
    


    
![png](output_2_2837.png)
    


    # of peaks : 0
    1420
    


    
![png](output_2_2839.png)
    


    # of peaks : 0
    1421
    


    
![png](output_2_2841.png)
    


    # of peaks : 0
    1422
    


    
![png](output_2_2843.png)
    


    # of peaks : 0
    1423
    


    
![png](output_2_2845.png)
    


    # of peaks : 0
    1424
    


    
![png](output_2_2847.png)
    


    # of peaks : 0
    1425
    


    
![png](output_2_2849.png)
    


    # of peaks : 0
    1426
    


    
![png](output_2_2851.png)
    


    # of peaks : 0
    1427
    


    
![png](output_2_2853.png)
    


    # of peaks : 0
    1428
    


    
![png](output_2_2855.png)
    


    # of peaks : 1
    1429
    


    
![png](output_2_2857.png)
    


    # of peaks : 0
    1430
    


    
![png](output_2_2859.png)
    


    # of peaks : 0
    1431
    


    
![png](output_2_2861.png)
    


    # of peaks : 0
    1432
    


    
![png](output_2_2863.png)
    


    # of peaks : 0
    1433
    


    
![png](output_2_2865.png)
    


    # of peaks : 0
    1434
    


    
![png](output_2_2867.png)
    


    # of peaks : 0
    1435
    


    
![png](output_2_2869.png)
    


    # of peaks : 0
    1436
    


    
![png](output_2_2871.png)
    


    # of peaks : 0
    1437
    


    
![png](output_2_2873.png)
    


    # of peaks : 0
    1438
    


    
![png](output_2_2875.png)
    


    # of peaks : 0
    1439
    


    
![png](output_2_2877.png)
    


    # of peaks : 0
    1440
    


    
![png](output_2_2879.png)
    


    # of peaks : 0
    1441
    


    
![png](output_2_2881.png)
    


    # of peaks : 1
    1442
    


    
![png](output_2_2883.png)
    


    # of peaks : 0
    1443
    


    
![png](output_2_2885.png)
    


    # of peaks : 0
    1444
    


    
![png](output_2_2887.png)
    


    # of peaks : 0
    1445
    


    
![png](output_2_2889.png)
    


    # of peaks : 0
    1446
    


    
![png](output_2_2891.png)
    


    # of peaks : 0
    1447
    


    
![png](output_2_2893.png)
    


    # of peaks : 0
    1448
    


    
![png](output_2_2895.png)
    


    # of peaks : 0
    1449
    


    
![png](output_2_2897.png)
    


    # of peaks : 1
    1450
    


    
![png](output_2_2899.png)
    


    # of peaks : 0
    1451
    


    
![png](output_2_2901.png)
    


    # of peaks : 0
    1452
    


    
![png](output_2_2903.png)
    


    # of peaks : 0
    1453
    


    
![png](output_2_2905.png)
    


    # of peaks : 0
    1454
    


    
![png](output_2_2907.png)
    


    # of peaks : 0
    1455
    


    
![png](output_2_2909.png)
    


    # of peaks : 0
    1456
    


    
![png](output_2_2911.png)
    


    # of peaks : 0
    1457
    


    
![png](output_2_2913.png)
    


    # of peaks : 0
    1458
    


    
![png](output_2_2915.png)
    


    # of peaks : 0
    1459
    


    
![png](output_2_2917.png)
    


    # of peaks : 0
    1460
    


    
![png](output_2_2919.png)
    


    # of peaks : 0
    1461
    


    
![png](output_2_2921.png)
    


    # of peaks : 1
    1462
    


    
![png](output_2_2923.png)
    


    # of peaks : 0
    1463
    


    
![png](output_2_2925.png)
    


    # of peaks : 0
    1464
    


    
![png](output_2_2927.png)
    


    # of peaks : 0
    1465
    


    
![png](output_2_2929.png)
    


    # of peaks : 0
    1466
    


    
![png](output_2_2931.png)
    


    # of peaks : 0
    1467
    


    
![png](output_2_2933.png)
    


    # of peaks : 1
    1468
    


    
![png](output_2_2935.png)
    


    # of peaks : 0
    1469
    


    
![png](output_2_2937.png)
    


    # of peaks : 1
    1470
    


    
![png](output_2_2939.png)
    


    # of peaks : 0
    1471
    


    
![png](output_2_2941.png)
    


    # of peaks : 0
    1472
    


    
![png](output_2_2943.png)
    


    # of peaks : 1
    1473
    


    
![png](output_2_2945.png)
    


    # of peaks : 0
    1474
    


    
![png](output_2_2947.png)
    


    # of peaks : 0
    1475
    


    
![png](output_2_2949.png)
    


    # of peaks : 0
    1476
    


    
![png](output_2_2951.png)
    


    # of peaks : 0
    1477
    


    
![png](output_2_2953.png)
    


    # of peaks : 0
    1478
    


    
![png](output_2_2955.png)
    


    # of peaks : 1
    1479
    


    
![png](output_2_2957.png)
    


    # of peaks : 1
    1480
    


    
![png](output_2_2959.png)
    


    # of peaks : 0
    1481
    


    
![png](output_2_2961.png)
    


    # of peaks : 0
    1482
    


    
![png](output_2_2963.png)
    


    # of peaks : 0
    1483
    


    
![png](output_2_2965.png)
    


    # of peaks : 0
    1484
    


    
![png](output_2_2967.png)
    


    # of peaks : 1
    1485
    


    
![png](output_2_2969.png)
    


    # of peaks : 0
    1486
    


    
![png](output_2_2971.png)
    


    # of peaks : 0
    1487
    


    
![png](output_2_2973.png)
    


    # of peaks : 0
    1488
    


    
![png](output_2_2975.png)
    


    # of peaks : 0
    1489
    


    
![png](output_2_2977.png)
    


    # of peaks : 1
    1490
    


    
![png](output_2_2979.png)
    


    # of peaks : 0
    1491
    


    
![png](output_2_2981.png)
    


    # of peaks : 1
    1492
    


    
![png](output_2_2983.png)
    


    # of peaks : 0
    1493
    


    
![png](output_2_2985.png)
    


    # of peaks : 0
    1494
    


    
![png](output_2_2987.png)
    


    # of peaks : 0
    1495
    


    
![png](output_2_2989.png)
    


    # of peaks : 0
    1496
    


    
![png](output_2_2991.png)
    


    # of peaks : 1
    1497
    


    
![png](output_2_2993.png)
    


    # of peaks : 1
    1498
    


    
![png](output_2_2995.png)
    


    # of peaks : 0
    1499
    


    
![png](output_2_2997.png)
    


    # of peaks : 0
    1500
    


    
![png](output_2_2999.png)
    


    # of peaks : 0
    1501
    


    
![png](output_2_3001.png)
    


    # of peaks : 0
    1502
    


    
![png](output_2_3003.png)
    


    # of peaks : 0
    1503
    


    
![png](output_2_3005.png)
    


    # of peaks : 0
    1504
    


    
![png](output_2_3007.png)
    


    # of peaks : 0
    1505
    


    
![png](output_2_3009.png)
    


    # of peaks : 0
    1506
    


    
![png](output_2_3011.png)
    


    # of peaks : 0
    1507
    


    
![png](output_2_3013.png)
    


    # of peaks : 0
    1508
    


    
![png](output_2_3015.png)
    


    # of peaks : 0
    1509
    


    
![png](output_2_3017.png)
    


    # of peaks : 1
    1510
    


    
![png](output_2_3019.png)
    


    # of peaks : 0
    1511
    


    
![png](output_2_3021.png)
    


    # of peaks : 0
    1512
    


    
![png](output_2_3023.png)
    


    # of peaks : 0
    1513
    


    
![png](output_2_3025.png)
    


    # of peaks : 0
    1514
    


    
![png](output_2_3027.png)
    


    # of peaks : 0
    1515
    


    
![png](output_2_3029.png)
    


    # of peaks : 0
    1516
    


    
![png](output_2_3031.png)
    


    # of peaks : 0
    1517
    


    
![png](output_2_3033.png)
    


    # of peaks : 0
    1518
    


    
![png](output_2_3035.png)
    


    # of peaks : 1
    1519
    


    
![png](output_2_3037.png)
    


    # of peaks : 0
    1520
    


    
![png](output_2_3039.png)
    


    # of peaks : 0
    1521
    


    
![png](output_2_3041.png)
    


    # of peaks : 0
    1522
    


    
![png](output_2_3043.png)
    


    # of peaks : 1
    1523
    


    
![png](output_2_3045.png)
    


    # of peaks : 0
    1524
    


    
![png](output_2_3047.png)
    


    # of peaks : 0
    1525
    


    
![png](output_2_3049.png)
    


    # of peaks : 0
    1526
    


    
![png](output_2_3051.png)
    


    # of peaks : 0
    1527
    


    
![png](output_2_3053.png)
    


    # of peaks : 0
    1528
    


    
![png](output_2_3055.png)
    


    # of peaks : 0
    1529
    


    
![png](output_2_3057.png)
    


    # of peaks : 0
    1530
    


    
![png](output_2_3059.png)
    


    # of peaks : 0
    1531
    


    
![png](output_2_3061.png)
    


    # of peaks : 0
    1532
    


    
![png](output_2_3063.png)
    


    # of peaks : 0
    1533
    


    
![png](output_2_3065.png)
    


    # of peaks : 0
    1534
    


    
![png](output_2_3067.png)
    


    # of peaks : 0
    1535
    


    
![png](output_2_3069.png)
    


    # of peaks : 0
    1536
    


    
![png](output_2_3071.png)
    


    # of peaks : 0
    1537
    


    
![png](output_2_3073.png)
    


    # of peaks : 1
    1538
    


    
![png](output_2_3075.png)
    


    # of peaks : 0
    1539
    


    
![png](output_2_3077.png)
    


    # of peaks : 1
    1540
    


    
![png](output_2_3079.png)
    


    # of peaks : 0
    1541
    


    
![png](output_2_3081.png)
    


    # of peaks : 0
    1542
    


    
![png](output_2_3083.png)
    


    # of peaks : 1
    1543
    


    
![png](output_2_3085.png)
    


    # of peaks : 0
    1544
    


    
![png](output_2_3087.png)
    


    # of peaks : 0
    1545
    


    
![png](output_2_3089.png)
    


    # of peaks : 0
    1546
    


    
![png](output_2_3091.png)
    


    # of peaks : 1
    1547
    


    
![png](output_2_3093.png)
    


    # of peaks : 1
    1548
    


    
![png](output_2_3095.png)
    


    # of peaks : 0
    1549
    


    
![png](output_2_3097.png)
    


    # of peaks : 0
    1550
    


    
![png](output_2_3099.png)
    


    # of peaks : 0
    1551
    


    
![png](output_2_3101.png)
    


    # of peaks : 0
    1552
    


    
![png](output_2_3103.png)
    


    # of peaks : 0
    1553
    


    
![png](output_2_3105.png)
    


    # of peaks : 0
    1554
    


    
![png](output_2_3107.png)
    


    # of peaks : 1
    1555
    


    
![png](output_2_3109.png)
    


    # of peaks : 0
    1556
    


    
![png](output_2_3111.png)
    


    # of peaks : 0
    1557
    


    
![png](output_2_3113.png)
    


    # of peaks : 0
    1558
    


    
![png](output_2_3115.png)
    


    # of peaks : 0
    1559
    


    
![png](output_2_3117.png)
    


    # of peaks : 0
    1560
    


    
![png](output_2_3119.png)
    


    # of peaks : 0
    1561
    


    
![png](output_2_3121.png)
    


    # of peaks : 0
    1562
    


    
![png](output_2_3123.png)
    


    # of peaks : 0
    1563
    


    
![png](output_2_3125.png)
    


    # of peaks : 0
    1564
    


    
![png](output_2_3127.png)
    


    # of peaks : 0
    1565
    


    
![png](output_2_3129.png)
    


    # of peaks : 0
    1566
    


    
![png](output_2_3131.png)
    


    # of peaks : 0
    1567
    


    
![png](output_2_3133.png)
    


    # of peaks : 0
    1568
    


    
![png](output_2_3135.png)
    


    # of peaks : 0
    1569
    


    
![png](output_2_3137.png)
    


    # of peaks : 0
    1570
    


    
![png](output_2_3139.png)
    


    # of peaks : 0
    1571
    


    
![png](output_2_3141.png)
    


    # of peaks : 0
    1572
    


    
![png](output_2_3143.png)
    


    # of peaks : 1
    1573
    


    
![png](output_2_3145.png)
    


    # of peaks : 0
    1574
    


    
![png](output_2_3147.png)
    


    # of peaks : 0
    1575
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-4-bb17a9f0344f> in <module>
         11     plt.plot((peaks+1)*0.02, x[peaks], "x")
         12     plt.vlines(x=(peaks+1)*0.02, ymin=contour_heights, ymax=x[peaks])
    ---> 13     plt.show()
         14     print('# of peaks : %d'%sum([1 for i in prominences if i >= -min(x)/3]))
         15     a += 1
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\pyplot.py in show(*args, **kwargs)
        351     """
        352     _warn_if_gui_out_of_main_thread()
    --> 353     return _backend_mod.show(*args, **kwargs)
        354 
        355 
    

    Z:\ai\anaconda\lib\site-packages\ipykernel\pylab\backend_inline.py in show(close, block)
         39     try:
         40         for figure_manager in Gcf.get_all_fig_managers():
    ---> 41             display(
         42                 figure_manager.canvas.figure,
         43                 metadata=_fetch_figure_metadata(figure_manager.canvas.figure)
    

    Z:\ai\anaconda\lib\site-packages\IPython\core\display.py in display(include, exclude, metadata, transient, display_id, *objs, **kwargs)
        311             publish_display_data(data=obj, metadata=metadata, **kwargs)
        312         else:
    --> 313             format_dict, md_dict = format(obj, include=include, exclude=exclude)
        314             if not format_dict:
        315                 # nothing to display (e.g. _ipython_display_ took over)
    

    Z:\ai\anaconda\lib\site-packages\IPython\core\formatters.py in format(self, obj, include, exclude)
        178             md = None
        179             try:
    --> 180                 data = formatter(obj)
        181             except:
        182                 # FIXME: log the exception
    

    <decorator-gen-2> in __call__(self, obj)
    

    Z:\ai\anaconda\lib\site-packages\IPython\core\formatters.py in catch_format_error(method, self, *args, **kwargs)
        222     """show traceback on failed format call"""
        223     try:
    --> 224         r = method(self, *args, **kwargs)
        225     except NotImplementedError:
        226         # don't warn on NotImplementedErrors
    

    Z:\ai\anaconda\lib\site-packages\IPython\core\formatters.py in __call__(self, obj)
        339                 pass
        340             else:
    --> 341                 return printer(obj)
        342             # Finally look for special method names
        343             method = get_real_method(obj, self.print_method)
    

    Z:\ai\anaconda\lib\site-packages\IPython\core\pylabtools.py in <lambda>(fig)
        246 
        247     if 'png' in formats:
    --> 248         png_formatter.for_type(Figure, lambda fig: print_figure(fig, 'png', **kwargs))
        249     if 'retina' in formats or 'png2x' in formats:
        250         png_formatter.for_type(Figure, lambda fig: retina_figure(fig, **kwargs))
    

    Z:\ai\anaconda\lib\site-packages\IPython\core\pylabtools.py in print_figure(fig, fmt, bbox_inches, **kwargs)
        130         FigureCanvasBase(fig)
        131 
    --> 132     fig.canvas.print_figure(bytes_io, **kw)
        133     data = bytes_io.getvalue()
        134     if fmt == 'svg':
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\backend_bases.py in print_figure(self, filename, dpi, facecolor, edgecolor, orientation, format, bbox_inches, pad_inches, bbox_extra_artists, backend, **kwargs)
       2208 
       2209             try:
    -> 2210                 result = print_method(
       2211                     filename,
       2212                     dpi=dpi,
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\backend_bases.py in wrapper(*args, **kwargs)
       1637             kwargs.pop(arg)
       1638 
    -> 1639         return func(*args, **kwargs)
       1640 
       1641     return wrapper
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\backends\backend_agg.py in print_png(self, filename_or_obj, metadata, pil_kwargs, *args)
        507             *metadata*, including the default 'Software' key.
        508         """
    --> 509         FigureCanvasAgg.draw(self)
        510         mpl.image.imsave(
        511             filename_or_obj, self.buffer_rgba(), format="png", origin="upper",
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\backends\backend_agg.py in draw(self)
        405              (self.toolbar._wait_cursor_for_draw_cm() if self.toolbar
        406               else nullcontext()):
    --> 407             self.figure.draw(self.renderer)
        408             # A GUI class may be need to update a window using this draw, so
        409             # don't forget to call the superclass.
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         39                 renderer.start_filter()
         40 
    ---> 41             return draw(artist, renderer, *args, **kwargs)
         42         finally:
         43             if artist.get_agg_filter() is not None:
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\figure.py in draw(self, renderer)
       1861 
       1862             self.patch.draw(renderer)
    -> 1863             mimage._draw_list_compositing_images(
       1864                 renderer, self, artists, self.suppressComposite)
       1865 
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\image.py in _draw_list_compositing_images(renderer, parent, artists, suppress_composite)
        129     if not_composite or not has_images:
        130         for a in artists:
    --> 131             a.draw(renderer)
        132     else:
        133         # Composite any adjacent images together
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         39                 renderer.start_filter()
         40 
    ---> 41             return draw(artist, renderer, *args, **kwargs)
         42         finally:
         43             if artist.get_agg_filter() is not None:
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\cbook\deprecation.py in wrapper(*inner_args, **inner_kwargs)
        409                          else deprecation_addendum,
        410                 **kwargs)
    --> 411         return func(*inner_args, **inner_kwargs)
        412 
        413     return wrapper
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\axes\_base.py in draw(self, renderer, inframe)
       2745             renderer.stop_rasterizing()
       2746 
    -> 2747         mimage._draw_list_compositing_images(renderer, self, artists)
       2748 
       2749         renderer.close_group('axes')
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\image.py in _draw_list_compositing_images(renderer, parent, artists, suppress_composite)
        129     if not_composite or not has_images:
        130         for a in artists:
    --> 131             a.draw(renderer)
        132     else:
        133         # Composite any adjacent images together
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         39                 renderer.start_filter()
         40 
    ---> 41             return draw(artist, renderer, *args, **kwargs)
         42         finally:
         43             if artist.get_agg_filter() is not None:
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\axis.py in draw(self, renderer, *args, **kwargs)
       1167 
       1168         for tick in ticks_to_draw:
    -> 1169             tick.draw(renderer)
       1170 
       1171         # scale up the axis label box to also find the neighbors, not
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         39                 renderer.start_filter()
         40 
    ---> 41             return draw(artist, renderer, *args, **kwargs)
         42         finally:
         43             if artist.get_agg_filter() is not None:
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\axis.py in draw(self, renderer)
        289         for artist in [self.gridline, self.tick1line, self.tick2line,
        290                        self.label1, self.label2]:
    --> 291             artist.draw(renderer)
        292         renderer.close_group(self.__name__)
        293         self.stale = False
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         39                 renderer.start_filter()
         40 
    ---> 41             return draw(artist, renderer, *args, **kwargs)
         42         finally:
         43             if artist.get_agg_filter() is not None:
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\text.py in draw(self, renderer)
        723                                           mtext=mtext)
        724                 else:
    --> 725                     textrenderer.draw_text(gc, x, y, clean_line,
        726                                            textobj._fontproperties, angle,
        727                                            ismath=ismath, mtext=mtext)
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\backends\backend_agg.py in draw_text(self, gc, x, y, s, prop, angle, ismath, mtext)
        193 
        194         flags = get_hinting_flag()
    --> 195         font = self._get_agg_font(prop)
        196 
        197         if font is None:
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\backends\backend_agg.py in _get_agg_font(self, prop)
        270         Get the font for text instance t, caching for efficiency
        271         """
    --> 272         fname = findfont(prop)
        273         font = get_font(fname)
        274 
    

    Z:\ai\anaconda\lib\site-packages\matplotlib\font_manager.py in findfont(self, prop, fontext, directory, fallback_to_default, rebuild_if_missing)
       1312             prop, fontext, directory, fallback_to_default, rebuild_if_missing,
       1313             rc_params)
    -> 1314         return os.path.realpath(filename)
       1315 
       1316     @lru_cache()
    

    Z:\ai\anaconda\lib\ntpath.py in realpath(path)
        645             path = join(cwd, path)
        646         try:
    --> 647             path = _getfinalpathname(path)
        648             initial_winerror = 0
        649         except OSError as ex:
    

    KeyboardInterrupt: 



```python
col = []
for i in test:
    x = np.array([0]+i.split(',')+[0]).astype('float64')
    peaks, _ = find_peaks(x)
    prominences = peak_prominences(x, peaks)[0]
    try: 
        if max(prominences) >= -min(x)*0.7:
            col.append(1)
        else:
            col.append(0)
    except:
        col.append(0)

df_new = pd.concat([df,pd.Series(col, name = 'pro_0.7')], axis = 1)
```




    '\n    try:\n        id = np.where(prominences == max(prominences))[0][0]\n        local_ht = x[peaks[id]]-min(x)\n        \n    except:\n        \n        local_ht = 0\n    if local_ht >= -min(x)*0.9:\n        col.append(1)\n    else:\n        col.append(0)\n'




```python

for i in range(df_new.shape[0]):
    if df_new.loc[i, 'pro_0.７'] != 0:
        print('Sub %s_pore %s, elong: %f' %(df_new.iloc[i, 0], df_new.iloc[i, 1], (df_new.iloc[i, 6]/df_new.iloc[i,7])))
        x = np.array([0]+df_new.iloc[i, 8].split(',')+[0]).astype('float64')
        line = np.array([0.02*(j+1) for j in range(x.size)])
        peaks, _ = find_peaks(x)
        prominences = peak_prominences(x, peaks)[0]
        
        plt.plot(line, x)
        id = np.where(prominences == max(prominences))[0][0]
        local_ht = x[peaks[id]]-prominences[id]
        plt.plot(((peaks+1)*0.02)[id], x[peaks][id], "x")
        plt.vlines(x=((peaks+1)*0.02)[id], ymin=local_ht, ymax=x[peaks][id])
        plt.show()
        print('Local height = %f, Total height = %f' %(prominences[id],-min(x)) )
```

    Sub 4_pore 140, elong: 10.864886
    


    
![png](output_4_1.png)
    


    Local height = 0.121000, Total height = 0.128000
    Sub 6_pore 29, elong: 1.808101
    


    
![png](output_4_3.png)
    


    Local height = 0.095000, Total height = 0.076000
    Sub 6_pore 44, elong: 14.090368
    


    
![png](output_4_5.png)
    


    Local height = 0.172000, Total height = 0.097000
    Sub 6_pore 210, elong: 3.059874
    


    
![png](output_4_7.png)
    


    Local height = 0.050000, Total height = 0.059000
    Sub 9_pore 44, elong: 3.722885
    


    
![png](output_4_9.png)
    


    Local height = 0.134000, Total height = 0.116000
    Sub 9_pore 61, elong: 8.544004
    


    
![png](output_4_11.png)
    


    Local height = 0.102000, Total height = 0.102000
    Sub 9_pore 131, elong: 3.269667
    


    
![png](output_4_13.png)
    


    Local height = 0.068000, Total height = 0.077000
    Sub 12_pore 123, elong: 1.933457
    


    
![png](output_4_15.png)
    


    Local height = 0.072999, Total height = 0.079000
    Sub 13_pore 35, elong: 2.329115
    


    
![png](output_4_17.png)
    


    Local height = 0.063000, Total height = 0.073000
    Sub 13_pore 120, elong: 2.607681
    


    
![png](output_4_19.png)
    


    Local height = 0.083999, Total height = 0.078000
    Sub 13_pore 251, elong: 1.374261
    


    
![png](output_4_21.png)
    


    Local height = 0.069999, Total height = 0.083000
    Sub 15_pore 261, elong: 13.346306
    


    
![png](output_4_23.png)
    


    Local height = 0.078000, Total height = 0.095000
    Sub 17_pore 46, elong: 3.813427
    


    
![png](output_4_25.png)
    


    Local height = 0.120000, Total height = 0.127000
    Sub 17_pore 166, elong: 3.000000
    


    
![png](output_4_27.png)
    


    Local height = 0.102000, Total height = 0.105000
    Sub 17_pore 180, elong: 2.812205
    


    
![png](output_4_29.png)
    


    Local height = 0.072000, Total height = 0.079000
    Sub 17_pore 214, elong: 4.829444
    


    
![png](output_4_31.png)
    


    Local height = 0.097000, Total height = 0.121000
    Sub 18_pore 47, elong: 7.103629
    


    
![png](output_4_33.png)
    


    Local height = 0.071000, Total height = 0.081000
    Sub 20_pore 124, elong: 2.920616
    


    
![png](output_4_35.png)
    


    Local height = 0.053000, Total height = 0.066000
    Sub 20_pore 282, elong: 2.978049
    


    
![png](output_4_37.png)
    


    Local height = 0.103999, Total height = 0.076000
    Sub 20_pore 300, elong: 6.789090
    


    
![png](output_4_39.png)
    


    Local height = 0.102000, Total height = 0.119999
    Sub 21_pore 226, elong: 3.733940
    


    
![png](output_4_41.png)
    


    Local height = 0.059000, Total height = 0.070000
    Sub 22_pore 59, elong: 7.325696
    


    
![png](output_4_43.png)
    


    Local height = 0.104000, Total height = 0.127000
    Sub 24_pore 58, elong: 5.224682
    


    
![png](output_4_45.png)
    


    Local height = 0.060999, Total height = 0.067000
    Sub 26_pore 350, elong: 9.056058
    


    
![png](output_4_47.png)
    


    Local height = 0.098000, Total height = 0.120000
    Sub 28_pore 53, elong: 1.169755
    


    
![png](output_4_49.png)
    


    Local height = 0.117000, Total height = 0.138000
    Sub 29_pore 171, elong: 7.001458
    


    
![png](output_4_51.png)
    


    Local height = 0.102999, Total height = 0.119000
    Sub 45_pore 323, elong: 4.362012
    


    
![png](output_4_53.png)
    


    Local height = 0.149000, Total height = 0.131000
    Sub 47_pore 85, elong: 1.389991
    


    
![png](output_4_55.png)
    


    Local height = 0.084000, Total height = 0.088000
    Sub 47_pore 90, elong: 3.096214
    


    
![png](output_4_57.png)
    


    Local height = 0.053000, Total height = 0.066000
    Sub 49_pore 202, elong: 10.911462
    


    
![png](output_4_59.png)
    


    Local height = 0.090000, Total height = 0.110000
    Sub 49_pore 266, elong: 2.795711
    


    
![png](output_4_61.png)
    


    Local height = 0.085000, Total height = 0.083000
    Sub 50_pore 293, elong: 3.205599
    


    
![png](output_4_63.png)
    


    Local height = 0.114000, Total height = 0.037001
    Sub 51_pore 209, elong: 5.954059
    


    
![png](output_4_65.png)
    


    Local height = 0.084000, Total height = 0.103000
    Sub 55_pore 170, elong: 6.977797
    


    
![png](output_4_67.png)
    


    Local height = 0.127000, Total height = 0.109000
    Sub 59_pore 209, elong: 7.544738
    


    
![png](output_4_69.png)
    


    Local height = 0.104000, Total height = 0.116000
    Sub 59_pore 214, elong: 6.533673
    


    
![png](output_4_71.png)
    


    Local height = 0.062000, Total height = 0.077000
    Sub 59_pore 333, elong: 7.037532
    


    
![png](output_4_73.png)
    


    Local height = 0.103000, Total height = 0.111000
    Sub 59_pore 351, elong: 4.043038
    


    
![png](output_4_75.png)
    


    Local height = 0.089000, Total height = 0.097000
    Sub 59_pore 353, elong: 5.743539
    


    
![png](output_4_77.png)
    


    Local height = 0.078000, Total height = 0.096000
    Sub 62_pore 185, elong: 13.332791
    


    
![png](output_4_79.png)
    


    Local height = 0.128000, Total height = 0.139000
    Sub 64_pore 48, elong: 4.797798
    


    
![png](output_4_81.png)
    


    Local height = 0.077000, Total height = 0.086000
    Sub 64_pore 256, elong: 2.432769
    


    
![png](output_4_83.png)
    


    Local height = 0.054000, Total height = 0.063000
    Sub 71_pore 26, elong: 5.458938
    


    
![png](output_4_85.png)
    


    Local height = 0.058000, Total height = 0.067000
    Sub 82_pore 164, elong: 2.249081
    


    
![png](output_4_87.png)
    


    Local height = 0.077000, Total height = 0.093000
    Sub 84_pore 22, elong: 2.223130
    


    
![png](output_4_89.png)
    


    Local height = 0.073000, Total height = 0.090000
    Sub 84_pore 166, elong: 1.739942
    


    
![png](output_4_91.png)
    


    Local height = 0.069999, Total height = 0.078000
    Sub 84_pore 205, elong: 19.419578
    


    
![png](output_4_93.png)
    


    Local height = 0.073000, Total height = 0.089999
    Sub 86_pore 64, elong: 6.044877
    


    
![png](output_4_95.png)
    


    Local height = 0.097999, Total height = 0.122000
    Sub 89_pore 208, elong: 4.385290
    


    
![png](output_4_97.png)
    


    Local height = 0.069000, Total height = 0.085000
    Sub 89_pore 227, elong: 7.937962
    


    
![png](output_4_99.png)
    


    Local height = 0.062999, Total height = 0.070000
    Sub 91_pore 23, elong: 6.800118
    


    
![png](output_4_101.png)
    


    Local height = 0.084999, Total height = 0.099000
    Sub 93_pore 209, elong: 9.221889
    


    
![png](output_4_103.png)
    


    Local height = 0.125000, Total height = 0.133000
    Sub 93_pore 303, elong: 7.775682
    


    
![png](output_4_105.png)
    


    Local height = 0.104000, Total height = 0.117000
    Sub 95_pore 50, elong: 28.344519
    


    
![png](output_4_107.png)
    


    Local height = 0.115000, Total height = 0.121000
    Sub 95_pore 64, elong: 55.327000
    


    
![png](output_4_109.png)
    


    Local height = 0.116000, Total height = 0.137000
    Sub 95_pore 132, elong: 5.500000
    


    
![png](output_4_111.png)
    


    Local height = 0.087000, Total height = 0.098000
    Sub 96_pore 29, elong: 5.603234
    


    
![png](output_4_113.png)
    


    Local height = 0.117000, Total height = 0.105000
    Sub 96_pore 153, elong: 9.006171
    


    
![png](output_4_115.png)
    


    Local height = 0.129000, Total height = 0.116000
    Sub 96_pore 168, elong: 56.998710
    


    
![png](output_4_117.png)
    


    Local height = 0.110000, Total height = 0.134000
    Sub 96_pore 171, elong: 3.955064
    


    
![png](output_4_119.png)
    


    Local height = 0.080000, Total height = 0.090000
    Sub 97_pore 6, elong: 12.219833
    


    
![png](output_4_121.png)
    


    Local height = 0.127000, Total height = 0.114000
    Sub 97_pore 29, elong: 6.841053
    


    
![png](output_4_123.png)
    


    Local height = 0.073000, Total height = 0.065000
    Sub 97_pore 124, elong: 7.700649
    


    
![png](output_4_125.png)
    


    Local height = 0.116999, Total height = 0.118000
    Sub 97_pore 140, elong: 2.148497
    


    
![png](output_4_127.png)
    


    Local height = 0.212000, Total height = 0.149000
    Sub 97_pore 148, elong: 12.281023
    


    
![png](output_4_129.png)
    


    Local height = 0.083000, Total height = 0.074000
    Sub 97_pore 155, elong: 2.414039
    


    
![png](output_4_131.png)
    


    Local height = 0.101000, Total height = 0.075000
    Sub 97_pore 211, elong: 3.530894
    


    
![png](output_4_133.png)
    


    Local height = 0.103999, Total height = 0.121000
    Sub 98_pore 141, elong: 18.954677
    


    
![png](output_4_135.png)
    


    Local height = 0.119000, Total height = 0.123000
    Sub 98_pore 201, elong: 4.234335
    


    
![png](output_4_137.png)
    


    Local height = 0.174999, Total height = 0.129000
    Sub 100_pore 149, elong: 2.013841
    


    
![png](output_4_139.png)
    


    Local height = 0.070000, Total height = 0.072000
    Sub 100_pore 191, elong: 3.158186
    


    
![png](output_4_141.png)
    


    Local height = 0.059000, Total height = 0.073000
    Sub 102_pore 45, elong: 23.759209
    


    
![png](output_4_143.png)
    


    Local height = 0.134000, Total height = 0.139000
    Sub 102_pore 64, elong: 13.899470
    


    
![png](output_4_145.png)
    


    Local height = 0.224000, Total height = 0.239000
    Sub 102_pore 84, elong: 42.939218
    


    
![png](output_4_147.png)
    


    Local height = 0.146000, Total height = 0.147000
    Sub 102_pore 91, elong: 20.087807
    


    
![png](output_4_149.png)
    


    Local height = 0.144000, Total height = 0.105000
    Sub 103_pore 217, elong: 5.978102
    


    
![png](output_4_151.png)
    


    Local height = 0.064000, Total height = 0.075000
    Sub 104_pore 109, elong: 8.649324
    


    
![png](output_4_153.png)
    


    Local height = 0.081000, Total height = 0.085000
    Sub 104_pore 322, elong: 11.150895
    


    
![png](output_4_155.png)
    


    Local height = 0.116000, Total height = 0.140000
    Sub 108_pore 75, elong: 17.475257
    


    
![png](output_4_157.png)
    


    Local height = 0.144000, Total height = 0.177000
    Sub 108_pore 179, elong: 9.357446
    


    
![png](output_4_159.png)
    


    Local height = 0.168000, Total height = 0.197000
    Sub 108_pore 198, elong: 15.237401
    


    
![png](output_4_161.png)
    


    Local height = 0.110999, Total height = 0.133000
    Sub 109_pore 25, elong: 1.661379
    


    
![png](output_4_163.png)
    


    Local height = 0.085000, Total height = 0.101000
    Sub 109_pore 226, elong: 18.951271
    


    
![png](output_4_165.png)
    


    Local height = 0.113000, Total height = 0.114000
    Sub 110_pore 278, elong: 22.090722
    


    
![png](output_4_167.png)
    


    Local height = 0.082999, Total height = 0.103000
    


```python
li = []
for s in sub:
#     li.append(sum(df_new.loc[df['Subject']==s]['new_merge2_0.9']))    
    li.append(sum(df_new.loc[df['Subject']==s]['pro_0.7'])/len(df_new.loc[df['Subject']==s]['pro_0.7'])*100)
```


```python
out = pd.concat([pd.Series(sub, name = 'sub'), pd.Series(li, name='pro_0.7')], axis = 1)
out
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sub</th>
      <th>pro_0.7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>104</td>
      <td>1.608579</td>
    </tr>
    <tr>
      <th>97</th>
      <td>105</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>98</th>
      <td>108</td>
      <td>3.103448</td>
    </tr>
    <tr>
      <th>99</th>
      <td>109</td>
      <td>2.061856</td>
    </tr>
    <tr>
      <th>100</th>
      <td>110</td>
      <td>0.317460</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 2 columns</p>
</div>




```python
out.to_excel("z:/result.xlsx", index = False)
```
