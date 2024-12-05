function test234(tasks)
%TEST234 test GxB_eWiseUnion

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    tasks = [ ] ;
end
if (isempty (tasks))
    tasks = {
    {'logical','first', 0,0,0,0 }, ... % 5 5
    {'logical','first', 0,0,0,1 }, ... % 6 11
    {'logical','first', 0,1,0,0 }, ... % 2 13
    {'logical','first', 0,1,0,1 }, ... % 1 14
    {'logical','first', 1,0,0,0 }, ... % 2 16
    {'logical','first', 1,1,0,0 }, ... % 2 18
    {'logical','first', 1,1,0,1 }, ... % 1 19
    {'logical','second', 0,0,0,0 }, ... % 1 20
    {'logical','pair', 0,0,0,0 }, ... % 1 21
    {'logical','plus', 0,0,0,0 }, ... % 1 22
    {'logical','minus', 0,0,0,0 }, ... % 1 23
    {'logical','times', 0,0,0,0 }, ... % 1 24
    {'logical','iseq', 0,0,0,0 }, ... % 1 25
    {'logical','pow', 0,0,0,0 }, ... % 1 26
    {'logical','pow', 0,0,0,1 }, ... % 3 29
    {'logical','isgt', 0,0,0,0 }, ... % 1 30
    {'logical','islt', 0,0,0,0 }, ... % 1 31
    {'logical','isle', 0,0,0,0 }, ... % 1 32
    {'int8','first', 0,0,0,0 }, ... % 1 33
    {'int8','second', 0,0,0,0 }, ... % 1 34
    {'int8','plus', 0,0,0,0 }, ... % 1 35
    {'int8','minus', 0,0,0,0 }, ... % 1 36
    {'int8','rminus', 0,0,0,0 }, ... % 1 37
    {'int8','times', 0,0,0,0 }, ... % 1 38
    {'int8','div', 0,0,0,0 }, ... % 1 39
    {'int8','rdiv', 0,0,0,0 }, ... % 1 40
    {'int8','iseq', 0,0,0,0 }, ... % 1 41
    {'int8','isne', 0,0,0,0 }, ... % 1 42
    {'int8','eq', 0,0,0,0 }, ... % 1 43
    {'int8','ne', 0,0,0,0 }, ... % 1 44
    {'int8','pow', 0,0,0,0 }, ... % 1 45
    {'int8','min', 0,0,0,0 }, ... % 1 46
    {'int8','max', 0,0,0,0 }, ... % 1 47
    {'int8','isgt', 0,0,0,0 }, ... % 1 48
    {'int8','islt', 0,0,0,0 }, ... % 1 49
    {'int8','isge', 0,0,0,0 }, ... % 1 50
    {'int8','isle', 0,0,0,0 }, ... % 1 51
    {'int8','gt', 0,0,0,0 }, ... % 1 52
    {'int8','lt', 0,0,0,0 }, ... % 1 53
    {'int8','ge', 0,0,0,0 }, ... % 1 54
    {'int8','le', 0,0,0,0 }, ... % 1 55
    {'int8','or', 0,0,0,0 }, ... % 1 56
    {'int8','and', 0,0,0,0 }, ... % 1 57
    {'int8','xor', 0,0,0,0 }, ... % 1 58
    {'int8','bor', 0,0,0,0 }, ... % 1 59
    {'int8','band', 0,0,0,0 }, ... % 1 60
    {'int8','bxor', 0,0,0,0 }, ... % 1 61
    {'int8','bxnor', 0,0,0,0 }, ... % 1 62
    {'int8','bget', 0,0,0,0 }, ... % 1 63
    {'int8','bset', 0,0,0,0 }, ... % 1 64
    {'int8','bclr', 0,0,0,0 }, ... % 1 65
    {'int16','first', 0,0,0,0 }, ... % 1 66
    {'int16','second', 0,0,0,0 }, ... % 1 67
    {'int16','plus', 0,0,0,0 }, ... % 1 68
    {'int16','minus', 0,0,0,0 }, ... % 1 69
    {'int16','rminus', 0,0,0,0 }, ... % 1 70
    {'int16','times', 0,0,0,0 }, ... % 1 71
    {'int16','div', 0,0,0,0 }, ... % 1 72
    {'int16','rdiv', 0,0,0,0 }, ... % 1 73
    {'int16','iseq', 0,0,0,0 }, ... % 1 74
    {'int16','isne', 0,0,0,0 }, ... % 1 75
    {'int16','eq', 0,0,0,0 }, ... % 1 76
    {'int16','ne', 0,0,0,0 }, ... % 1 77
    {'int16','pow', 0,0,0,0 }, ... % 1 78
    {'int16','min', 0,0,0,0 }, ... % 1 79
    {'int16','max', 0,0,0,0 }, ... % 1 80
    {'int16','isgt', 0,0,0,0 }, ... % 1 81
    {'int16','islt', 0,0,0,0 }, ... % 1 82
    {'int16','isge', 0,0,0,0 }, ... % 1 83
    {'int16','isle', 0,0,0,0 }, ... % 1 84
    {'int16','gt', 0,0,0,0 }, ... % 1 85
    {'int16','lt', 0,0,0,0 }, ... % 1 86
    {'int16','ge', 0,0,0,0 }, ... % 1 87
    {'int16','le', 0,0,0,0 }, ... % 1 88
    {'int16','or', 0,0,0,0 }, ... % 1 89
    {'int16','and', 0,0,0,0 }, ... % 1 90
    {'int16','xor', 0,0,0,0 }, ... % 1 91
    {'int16','bor', 0,0,0,0 }, ... % 1 92
    {'int16','band', 0,0,0,0 }, ... % 1 93
    {'int16','bxor', 0,0,0,0 }, ... % 1 94
    {'int16','bxnor', 0,0,0,0 }, ... % 1 95
    {'int16','bget', 0,0,0,0 }, ... % 1 96
    {'int16','bset', 0,0,0,0 }, ... % 1 97
    {'int16','bclr', 0,0,0,0 }, ... % 1 98
    {'int32','first', 0,0,0,0 }, ... % 1 99
    {'int32','second', 0,0,0,0 }, ... % 1 100
    {'int32','plus', 0,0,0,0 }, ... % 1 101
    {'int32','minus', 0,0,0,0 }, ... % 1 102
    {'int32','rminus', 0,0,0,0 }, ... % 1 103
    {'int32','times', 0,0,0,0 }, ... % 1 104
    {'int32','div', 0,0,0,0 }, ... % 1 105
    {'int32','rdiv', 0,0,0,0 }, ... % 1 106
    {'int32','iseq', 0,0,0,0 }, ... % 1 107
    {'int32','isne', 0,0,0,0 }, ... % 1 108
    {'int32','eq', 0,0,0,0 }, ... % 1 109
    {'int32','ne', 0,0,0,0 }, ... % 1 110
    {'int32','pow', 0,0,0,0 }, ... % 1 111
    {'int32','min', 0,0,0,0 }, ... % 1 112
    {'int32','max', 0,0,0,0 }, ... % 1 113
    {'int32','isgt', 0,0,0,0 }, ... % 1 114
    {'int32','islt', 0,0,0,0 }, ... % 1 115
    {'int32','isge', 0,0,0,0 }, ... % 1 116
    {'int32','isle', 0,0,0,0 }, ... % 1 117
    {'int32','gt', 0,0,0,0 }, ... % 1 118
    {'int32','lt', 0,0,0,0 }, ... % 1 119
    {'int32','ge', 0,0,0,0 }, ... % 1 120
    {'int32','le', 0,0,0,0 }, ... % 1 121
    {'int32','or', 0,0,0,0 }, ... % 1 122
    {'int32','and', 0,0,0,0 }, ... % 1 123
    {'int32','xor', 0,0,0,0 }, ... % 1 124
    {'int32','bor', 0,0,0,0 }, ... % 1 125
    {'int32','band', 0,0,0,0 }, ... % 1 126
    {'int32','bxor', 0,0,0,0 }, ... % 1 127
    {'int32','bxnor', 0,0,0,0 }, ... % 1 128
    {'int32','bget', 0,0,0,0 }, ... % 1 129
    {'int32','bset', 0,0,0,0 }, ... % 1 130
    {'int32','bclr', 0,0,0,0 }, ... % 1 131
    {'int32','firsti', 0,0,0,0 }, ... % 1 132
    {'int64','first', 0,0,0,0 }, ... % 1 133
    {'int64','second', 0,0,0,0 }, ... % 1 134
    {'int64','plus', 0,0,0,0 }, ... % 1 135
    {'int64','minus', 0,0,0,0 }, ... % 1 136
    {'int64','rminus', 0,0,0,0 }, ... % 1 137
    {'int64','times', 0,0,0,0 }, ... % 1 138
    {'int64','div', 0,0,0,0 }, ... % 1 139
    {'int64','rdiv', 0,0,0,0 }, ... % 1 140
    {'int64','iseq', 0,0,0,0 }, ... % 1 141
    {'int64','isne', 0,0,0,0 }, ... % 1 142
    {'int64','eq', 0,0,0,0 }, ... % 1 143
    {'int64','ne', 0,0,0,0 }, ... % 1 144
    {'int64','pow', 0,0,0,0 }, ... % 1 145
    {'int64','min', 0,0,0,0 }, ... % 1 146
    {'int64','max', 0,0,0,0 }, ... % 1 147
    {'int64','isgt', 0,0,0,0 }, ... % 1 148
    {'int64','islt', 0,0,0,0 }, ... % 1 149
    {'int64','isge', 0,0,0,0 }, ... % 1 150
    {'int64','isle', 0,0,0,0 }, ... % 1 151
    {'int64','gt', 0,0,0,0 }, ... % 1 152
    {'int64','lt', 0,0,0,0 }, ... % 1 153
    {'int64','ge', 0,0,0,0 }, ... % 1 154
    {'int64','le', 0,0,0,0 }, ... % 1 155
    {'int64','or', 0,0,0,0 }, ... % 1 156
    {'int64','and', 0,0,0,0 }, ... % 1 157
    {'int64','xor', 0,0,0,0 }, ... % 1 158
    {'int64','bor', 0,0,0,0 }, ... % 1 159
    {'int64','band', 0,0,0,0 }, ... % 1 160
    {'int64','bxor', 0,0,0,0 }, ... % 1 161
    {'int64','bxnor', 0,0,0,0 }, ... % 1 162
    {'int64','bget', 0,0,0,0 }, ... % 1 163
    {'int64','bset', 0,0,0,0 }, ... % 1 164
    {'int64','bclr', 0,0,0,0 }, ... % 1 165
    {'uint8','first', 0,0,0,0 }, ... % 1 166
    {'uint8','second', 0,0,0,0 }, ... % 1 167
    {'uint8','plus', 0,0,0,0 }, ... % 1 168
    {'uint8','minus', 0,0,0,0 }, ... % 1 169
    {'uint8','rminus', 0,0,0,0 }, ... % 1 170
    {'uint8','times', 0,0,0,0 }, ... % 1 171
    {'uint8','div', 0,0,0,0 }, ... % 1 172
    {'uint8','rdiv', 0,0,0,0 }, ... % 1 173
    {'uint8','iseq', 0,0,0,0 }, ... % 1 174
    {'uint8','isne', 0,0,0,0 }, ... % 1 175
    {'uint8','eq', 0,0,0,0 }, ... % 1 176
    {'uint8','ne', 0,0,0,0 }, ... % 1 177
    {'uint8','pow', 0,0,0,0 }, ... % 1 178
    {'uint8','min', 0,0,0,0 }, ... % 1 179
    {'uint8','max', 0,0,0,0 }, ... % 1 180
    {'uint8','isgt', 0,0,0,0 }, ... % 1 181
    {'uint8','islt', 0,0,0,0 }, ... % 1 182
    {'uint8','isge', 0,0,0,0 }, ... % 1 183
    {'uint8','isle', 0,0,0,0 }, ... % 1 184
    {'uint8','gt', 0,0,0,0 }, ... % 1 185
    {'uint8','lt', 0,0,0,0 }, ... % 1 186
    {'uint8','ge', 0,0,0,0 }, ... % 1 187
    {'uint8','le', 0,0,0,0 }, ... % 1 188
    {'uint8','or', 0,0,0,0 }, ... % 1 189
    {'uint8','and', 0,0,0,0 }, ... % 1 190
    {'uint8','xor', 0,0,0,0 }, ... % 1 191
    {'uint8','bor', 0,0,0,0 }, ... % 1 192
    {'uint8','band', 0,0,0,0 }, ... % 1 193
    {'uint8','bxor', 0,0,0,0 }, ... % 1 194
    {'uint8','bxnor', 0,0,0,0 }, ... % 1 195
    {'uint8','bget', 0,0,0,0 }, ... % 1 196
    {'uint8','bset', 0,0,0,0 }, ... % 1 197
    {'uint8','bclr', 0,0,0,0 }, ... % 1 198
    {'uint16','first', 0,0,0,0 }, ... % 1 199
    {'uint16','second', 0,0,0,0 }, ... % 1 200
    {'uint16','plus', 0,0,0,0 }, ... % 1 201
    {'uint16','minus', 0,0,0,0 }, ... % 1 202
    {'uint16','rminus', 0,0,0,0 }, ... % 1 203
    {'uint16','times', 0,0,0,0 }, ... % 1 204
    {'uint16','div', 0,0,0,0 }, ... % 1 205
    {'uint16','rdiv', 0,0,0,0 }, ... % 1 206
    {'uint16','iseq', 0,0,0,0 }, ... % 1 207
    {'uint16','isne', 0,0,0,0 }, ... % 1 208
    {'uint16','eq', 0,0,0,0 }, ... % 1 209
    {'uint16','ne', 0,0,0,0 }, ... % 1 210
    {'uint16','pow', 0,0,0,0 }, ... % 1 211
    {'uint16','min', 0,0,0,0 }, ... % 1 212
    {'uint16','max', 0,0,0,0 }, ... % 1 213
    {'uint16','isgt', 0,0,0,0 }, ... % 1 214
    {'uint16','islt', 0,0,0,0 }, ... % 1 215
    {'uint16','isge', 0,0,0,0 }, ... % 1 216
    {'uint16','isle', 0,0,0,0 }, ... % 1 217
    {'uint16','gt', 0,0,0,0 }, ... % 1 218
    {'uint16','lt', 0,0,0,0 }, ... % 1 219
    {'uint16','ge', 0,0,0,0 }, ... % 1 220
    {'uint16','le', 0,0,0,0 }, ... % 1 221
    {'uint16','or', 0,0,0,0 }, ... % 1 222
    {'uint16','and', 0,0,0,0 }, ... % 1 223
    {'uint16','xor', 0,0,0,0 }, ... % 1 224
    {'uint16','bor', 0,0,0,0 }, ... % 1 225
    {'uint16','band', 0,0,0,0 }, ... % 1 226
    {'uint16','bxor', 0,0,0,0 }, ... % 1 227
    {'uint16','bxnor', 0,0,0,0 }, ... % 1 228
    {'uint16','bget', 0,0,0,0 }, ... % 1 229
    {'uint16','bset', 0,0,0,0 }, ... % 1 230
    {'uint16','bclr', 0,0,0,0 }, ... % 1 231
    {'uint32','first', 0,0,0,0 }, ... % 1 232
    {'uint32','second', 0,0,0,0 }, ... % 1 233
    {'uint32','plus', 0,0,0,0 }, ... % 1 234
    {'uint32','minus', 0,0,0,0 }, ... % 1 235
    {'uint32','rminus', 0,0,0,0 }, ... % 1 236
    {'uint32','times', 0,0,0,0 }, ... % 1 237
    {'uint32','div', 0,0,0,0 }, ... % 1 238
    {'uint32','rdiv', 0,0,0,0 }, ... % 1 239
    {'uint32','iseq', 0,0,0,0 }, ... % 1 240
    {'uint32','isne', 0,0,0,0 }, ... % 1 241
    {'uint32','eq', 0,0,0,0 }, ... % 1 242
    {'uint32','ne', 0,0,0,0 }, ... % 1 243
    {'uint32','pow', 0,0,0,0 }, ... % 1 244
    {'uint32','min', 0,0,0,0 }, ... % 1 245
    {'uint32','max', 0,0,0,0 }, ... % 1 246
    {'uint32','isgt', 0,0,0,0 }, ... % 1 247
    {'uint32','islt', 0,0,0,0 }, ... % 1 248
    {'uint32','isge', 0,0,0,0 }, ... % 1 249
    {'uint32','isle', 0,0,0,0 }, ... % 1 250
    {'uint32','gt', 0,0,0,0 }, ... % 1 251
    {'uint32','lt', 0,0,0,0 }, ... % 1 252
    {'uint32','ge', 0,0,0,0 }, ... % 1 253
    {'uint32','le', 0,0,0,0 }, ... % 1 254
    {'uint32','or', 0,0,0,0 }, ... % 1 255
    {'uint32','and', 0,0,0,0 }, ... % 1 256
    {'uint32','xor', 0,0,0,0 }, ... % 1 257
    {'uint32','bor', 0,0,0,0 }, ... % 1 258
    {'uint32','band', 0,0,0,0 }, ... % 1 259
    {'uint32','bxor', 0,0,0,0 }, ... % 1 260
    {'uint32','bxnor', 0,0,0,0 }, ... % 1 261
    {'uint32','bget', 0,0,0,0 }, ... % 1 262
    {'uint32','bset', 0,0,0,0 }, ... % 1 263
    {'uint32','bclr', 0,0,0,0 }, ... % 1 264
    {'uint64','first', 0,0,0,0 }, ... % 1 265
    {'uint64','second', 0,0,0,0 }, ... % 1 266
    {'uint64','plus', 0,0,0,0 }, ... % 1 267
    {'uint64','minus', 0,0,0,0 }, ... % 1 268
    {'uint64','rminus', 0,0,0,0 }, ... % 1 269
    {'uint64','times', 0,0,0,0 }, ... % 1 270
    {'uint64','div', 0,0,0,0 }, ... % 1 271
    {'uint64','rdiv', 0,0,0,0 }, ... % 1 272
    {'uint64','iseq', 0,0,0,0 }, ... % 1 273
    {'uint64','isne', 0,0,0,0 }, ... % 1 274
    {'uint64','eq', 0,0,0,0 }, ... % 1 275
    {'uint64','ne', 0,0,0,0 }, ... % 1 276
    {'uint64','pow', 0,0,0,0 }, ... % 1 277
    {'uint64','min', 0,0,0,0 }, ... % 1 278
    {'uint64','max', 0,0,0,0 }, ... % 1 279
    {'uint64','isgt', 0,0,0,0 }, ... % 1 280
    {'uint64','islt', 0,0,0,0 }, ... % 1 281
    {'uint64','isge', 0,0,0,0 }, ... % 1 282
    {'uint64','isle', 0,0,0,0 }, ... % 1 283
    {'uint64','gt', 0,0,0,0 }, ... % 1 284
    {'uint64','lt', 0,0,0,0 }, ... % 1 285
    {'uint64','ge', 0,0,0,0 }, ... % 1 286
    {'uint64','le', 0,0,0,0 }, ... % 1 287
    {'uint64','or', 0,0,0,0 }, ... % 1 288
    {'uint64','and', 0,0,0,0 }, ... % 1 289
    {'uint64','xor', 0,0,0,0 }, ... % 1 290
    {'uint64','bor', 0,0,0,0 }, ... % 1 291
    {'uint64','band', 0,0,0,0 }, ... % 1 292
    {'uint64','bxor', 0,0,0,0 }, ... % 1 293
    {'uint64','bxnor', 0,0,0,0 }, ... % 1 294
    {'uint64','bget', 0,0,0,0 }, ... % 1 295
    {'uint64','bset', 0,0,0,0 }, ... % 1 296
    {'uint64','bclr', 0,0,0,0 }, ... % 1 297
    {'single','first', 0,0,0,0 }, ... % 1 298
    {'single','second', 0,0,0,0 }, ... % 1 299
    {'single','plus', 0,0,0,0 }, ... % 1 300
    {'single','minus', 0,0,0,0 }, ... % 1 301
    {'single','rminus', 0,0,0,0 }, ... % 1 302
    {'single','times', 0,0,0,0 }, ... % 1 303
    {'single','div', 0,0,0,0 }, ... % 1 304
    {'single','rdiv', 0,0,0,0 }, ... % 1 305
    {'single','iseq', 0,0,0,0 }, ... % 1 306
    {'single','isne', 0,0,0,0 }, ... % 1 307
    {'single','eq', 0,0,0,0 }, ... % 1 308
    {'single','ne', 0,0,0,0 }, ... % 1 309
    {'single','pow', 0,0,0,0 }, ... % 1 310
    {'single','min', 0,0,0,0 }, ... % 1 311
    {'single','max', 0,0,0,0 }, ... % 1 312
    {'single','isgt', 0,0,0,0 }, ... % 1 313
    {'single','islt', 0,0,0,0 }, ... % 1 314
    {'single','isge', 0,0,0,0 }, ... % 1 315
    {'single','isle', 0,0,0,0 }, ... % 1 316
    {'single','gt', 0,0,0,0 }, ... % 1 317
    {'single','lt', 0,0,0,0 }, ... % 1 318
    {'single','ge', 0,0,0,0 }, ... % 1 319
    {'single','le', 0,0,0,0 }, ... % 1 320
    {'single','or', 0,0,0,0 }, ... % 1 321
    {'single','and', 0,0,0,0 }, ... % 1 322
    {'single','xor', 0,0,0,0 }, ... % 1 323
    {'single','atan2', 0,0,0,0 }, ... % 1 324
    {'single','hypot', 0,0,0,0 }, ... % 1 325
    {'single','fmod', 0,0,0,0 }, ... % 1 326
    {'single','remainder', 0,0,0,0 }, ... % 1 327
    {'single','ldexp', 0,0,0,0 }, ... % 1 328
    {'single','copysign', 0,0,0,0 }, ... % 1 329
    {'single','cmplx', 0,0,0,0 }, ... % 1 330
    {'double','minus', 0,0,0,0 }, ... % 1 331
    {'double','rminus', 0,0,0,0 }, ... % 1 332
    {'double','times', 0,0,0,0 }, ... % 1 333
    {'double','div', 0,0,0,0 }, ... % 1 334
    {'double','rdiv', 0,0,0,0 }, ... % 1 335
    {'double','iseq', 0,0,0,0 }, ... % 1 336
    {'double','isne', 0,0,0,0 }, ... % 1 337
    {'double','eq', 0,0,0,0 }, ... % 1 338
    {'double','ne', 0,0,0,0 }, ... % 1 339
    {'double','pow', 0,0,0,0 }, ... % 1 340
    {'double','min', 0,0,0,0 }, ... % 1 341
    {'double','max', 0,0,0,0 }, ... % 1 342
    {'double','isgt', 0,0,0,0 }, ... % 1 343
    {'double','islt', 0,0,0,0 }, ... % 1 344
    {'double','isge', 0,0,0,0 }, ... % 1 345
    {'double','isle', 0,0,0,0 }, ... % 1 346
    {'double','gt', 0,0,0,0 }, ... % 1 347
    {'double','lt', 0,0,0,0 }, ... % 1 348
    {'double','ge', 0,0,0,0 }, ... % 1 349
    {'double','le', 0,0,0,0 }, ... % 1 350
    {'double','or', 0,0,0,0 }, ... % 1 351
    {'double','and', 0,0,0,0 }, ... % 1 352
    {'double','xor', 0,0,0,0 }, ... % 1 353
    {'double','atan2', 0,0,0,0 }, ... % 1 354
    {'double','hypot', 0,0,0,0 }, ... % 1 355
    {'double','fmod', 0,0,0,0 }, ... % 1 356
    {'double','remainder', 0,0,0,0 }, ... % 1 357
    {'double','ldexp', 0,0,0,0 }, ... % 1 358
    {'double','copysign', 0,0,0,0 }, ... % 1 359
    {'double','cmplx', 0,0,0,0 }, ... % 1 360
    {'single complex','plus', 0,0,0,0 }, ... % 1 361
    {'single complex','minus', 0,0,0,0 }, ... % 1 362
    {'single complex','rminus', 0,0,0,0 }, ... % 1 363
    {'single complex','times', 0,0,0,0 }, ... % 1 364
    {'single complex','div', 0,0,0,0 }, ... % 1 365
    {'single complex','rdiv', 0,0,0,0 }, ... % 1 366
    {'single complex','iseq', 0,0,0,0 }, ... % 1 367
    {'single complex','isne', 0,0,0,0 }, ... % 1 368
    {'single complex','eq', 0,0,0,0 }, ... % 1 369
    {'single complex','ne', 0,0,0,0 }, ... % 1 370
    {'single complex','pow', 0,0,0,0 }, ... % 1 371
    {'double complex','first', 0,0,0,0 }, ... % 1 372
    {'double complex','second', 0,0,0,0 }, ... % 1 373
    {'double complex','plus', 0,0,0,0 }, ... % 1 374
    {'double complex','minus', 0,0,0,0 }, ... % 1 375
    {'double complex','rminus', 0,0,0,0 }, ... % 1 376
    {'double complex','times', 0,0,0,0 }, ... % 1 377
    {'double complex','div', 0,0,0,0 }, ... % 1 378
    {'double complex','rdiv', 0,0,0,0 }, ... % 1 379
    {'double complex','iseq', 0,0,0,0 }, ... % 1 380
    {'double complex','isne', 0,0,0,0 }, ... % 1 381
    {'double complex','eq', 0,0,0,0 }, ... % 1 382
    {'double complex','ne', 0,0,0,0 }, ... % 1 383
    {'double complex','pow', 0,0,0,0 }} ; % 1 384
end

ntasks = length (tasks) ;
fprintf ('test234 -----------tests of GxB_eWiseUnion (tasks: %d)\n', ntasks) ;

m = 5 ;
n = 5 ;

rng ('default') ;

dnn = struct ;
dnn_notM = struct ('mask', 'complement') ;

Amat2 = sparse (2 * sprand (m,n, 0.8)) ;
Bmat2 = sparse (2 * sprand (m,n, 0.8)) ;
Cmat2 = sparse (2 * sprand (m,n, 0.8)) ;
w2 = sparse (2 * sprand (m,1, 0.8)) ;
uvec2 = sparse (2 * sprand (m,1, 0.8)) ;
vvec2 = sparse (2 * sprand (m,1, 0.8)) ;

Amat = sparse (100 * sprandn (m,n, 0.8)) ;
Bmat = sparse (100 * sprandn (m,n, 0.8)) ;
Cmat = sparse (100 * sprandn (m,n, 0.8)) ;
w = sparse (100 * sprandn (m,1, 0.8)) ;
uvec = sparse (100 * sprandn (m,1, 0.8)) ;
vvec = sparse (100 * sprandn (m,1, 0.8)) ;

Maskmat = sprandn (m,n,0.9) ~= 0 ;
maskvec = sprandn (m,1,0.9) ~= 0 ;

% create a very sparse matrix mask
Maskmat2 = sparse (m,n) ;
T = Amat .* Bmat ;
[i j x] = find (T) ;
if (length (i) > 0)
    Maskmat2 (i(1), j(1)) = 1 ;
end
T = (Amat ~= 0) & (Bmat == 0) ;
[i j x] = find (T) ;
if (length (i) > 0)
    Maskmat2 (i(1), j(1)) = 1 ;
end
T = (Amat == 0) & (Bmat ~= 0) ;
[i j x] = find (T) ;
if (length (i) > 0)
    Maskmat2 (i(1), j(1)) = 1 ;
end
clear T i j x

% create a very sparse vector mask
maskvec2 = sparse (m,1) ;
T = uvec .* vvec ;
[i j x] = find (T) ;
if (length (i) > 0)
    maskvec2 (i(1), j(1)) = 1 ;
end
T = (uvec ~= 0) & (vvec == 0) ;
[i j x] = find (T) ;
if (length (i) > 0)
    maskvec2 (i(1), j(1)) = 1 ;
end
T = (uvec == 0) & (vvec ~= 0) ;
[i j x] = find (T) ;
if (length (i) > 0)
    maskvec2 (i(1), j(1)) = 1 ;
end
clear T i j x

A_is_csc   = 0 ;
B_is_csc   = 0 ;
C_is_csc   = 0 ;

M_is_very_sparse = 0 ;
M_is_csc   = 0 ;

track_coverage = false ;
if (track_coverage)
    global GraphBLAS_grbcov
    track_coverage = ~isempty (GraphBLAS_grbcov) ;
    clast = sum (GraphBLAS_grbcov > 0) ;
    cfirst = clast ;
end

for k1 = 1:ntasks
    task = tasks {k1} ;
    type = task {1} ;
    binop = task {2} ;
    A_sparsity_control = task {3} ;
    B_sparsity_control = task {4} ;
    C_sparsity_control = task {5} ;
    M_sparsity_control = task {6} ;

    op.opname = binop ;
    op.optype = type ;

    if (test_contains (type, 'single'))
        tol = 1e-5 ;
    elseif (test_contains (type, 'double'))
        tol = 1e-12 ;
    else
        tol = 0 ;
    end

    try
        GB_spec_operator (op) ;
    catch
        continue ;
    end

    if (A_sparsity_control == 0)
        A_is_hyper = 0 ; % not hyper
        A_sparsity = 1 ; % sparse
    else
        A_is_hyper = 0 ; % not hyper
        A_sparsity = 4 ; % bitmap
    end

    if (B_sparsity_control == 0)
        B_is_hyper = 0 ; % not hyper
        B_sparsity = 1 ; % sparse
    else
        B_is_hyper = 0 ; % not hyper
        B_sparsity = 4 ; % bitmap
    end

    if (C_sparsity_control == 0)
        C_is_hyper = 0 ; % not hyper
        C_sparsity = 1 ; % sparse
    else
        C_is_hyper = 0 ; % not hyper
        C_sparsity = 4 ; % bitmap
    end

    clear A B C u v

    if (isequal (binop, 'pow'))
        A.matrix = Amat2 ;
        B.matrix = Bmat2 ;
        C.matrix = Cmat2 ;
        u.matrix = uvec2 ;
        v.matrix = vvec2 ;
    else
        A.matrix = Amat ;
        B.matrix = Bmat ;
        C.matrix = Cmat ;
        u.matrix = uvec ;
        v.matrix = vvec ;
    end

    A.is_hyper = A_is_hyper ;
    A.is_csc   = A_is_csc   ;
    A.sparsity = A_sparsity ;
    A.class = op.optype ;
    a0 = GB_mex_cast (1, op.optype) ;

    B.is_hyper = B_is_hyper ;
    B.sparsity = B_sparsity ;
    B.is_csc   = B_is_csc   ;
    B.class = op.optype ;
    b0 = GB_mex_cast (2, op.optype) ;

    C.is_hyper = C_is_hyper ;
    C.is_csc   = C_is_csc   ;
    C.sparsity = C_sparsity ;

    u.is_csc = true ;
    u.class = op.optype ;
    u0 = GB_mex_cast (1, op.optype) ;

    v.is_csc = true ;
    v.class = op.optype ;
    v0 = GB_mex_cast (2, op.optype) ;

    %---------------------------------------
    % A+B
    %---------------------------------------

    C0 = GB_spec_Matrix_eWiseUnion (C, [ ], [ ], op, A, a0, B, b0, dnn) ;
    C1 = GB_mex_Matrix_eWiseUnion  (C, [ ], [ ], op, A, a0, B, b0, dnn) ;
    GB_spec_compare (C0, C1, 0, tol) ;

    w0 = GB_spec_Vector_eWiseUnion (w, [ ], [ ], op, u, u0, v, v0, dnn) ;
    w1 = GB_mex_Vector_eWiseUnion  (w, [ ], [ ], op, u, u0, v, v0, dnn) ;
    GB_spec_compare (w0, w1, 0, tol) ;

    %-----------------------------------------------
    % with mask
    %-----------------------------------------------

    clear Mask mask
    if (M_is_very_sparse)
        Mask.matrix = Maskmat2 ;
        mask.matrix = maskvec2 ;
    else
        Mask.matrix = Maskmat ;
        mask.matrix = maskvec ;
    end

    if (M_sparsity_control == 0)
        M_is_hyper = 0 ; % not hyper
        M_sparsity = 1 ; % sparse
    else
        M_is_hyper = 0 ; % not hyper
        M_sparsity = 4 ; % bitmap
    end

    Mask.is_hyper = M_is_hyper ;
    Mask.sparsity = M_sparsity ;
    Mask.is_csc   = M_is_csc   ;
    mask.is_csc = true ;

    %---------------------------------------
    % A+B, with mask
    %---------------------------------------

    C0 = GB_spec_Matrix_eWiseUnion (C, Mask, [ ], op, A, a0, B, b0, dnn) ;
    C1 = GB_mex_Matrix_eWiseUnion  (C, Mask, [ ], op, A, a0, B, b0, dnn) ;
    GB_spec_compare (C0, C1, 0, tol) ;

    w0 = GB_spec_Vector_eWiseUnion (w, mask, [ ], op, u, u0, v, v0, dnn) ;
    w1 = GB_mex_Vector_eWiseUnion  (w, mask, [ ], op, u, u0, v, v0, dnn) ;
    GB_spec_compare (w0, w1, 0, tol) ;

    %---------------------------------------
    % A+B, with mask complemented
    %---------------------------------------

    C0 = GB_spec_Matrix_eWiseUnion (C, Mask, [ ], op, A, a0, B, b0, dnn_notM) ;
    C1 = GB_mex_Matrix_eWiseUnion  (C, Mask, [ ], op, A, a0, B, b0, dnn_notM) ;
    GB_spec_compare (C0, C1, 0, tol) ;

    w0 = GB_spec_Vector_eWiseUnion (w, mask, [ ], op, u, u0, v, v0, dnn_notM) ;
    w1 = GB_mex_Vector_eWiseUnion  (w, mask, [ ], op, u, u0, v, v0, dnn_notM) ;
    GB_spec_compare (w0, w1, 0, tol) ;

    if (track_coverage)
        c = sum (GraphBLAS_grbcov > 0) ;
        d = c - clast ;
        if (d > 0)
            fprintf ('{''%s'',''%s'', %d,%d,%d,%d }, ... %% %d %d\n', ...
                type, binop, A_sparsity_control, B_sparsity_control, ...
                C_sparsity_control, M_sparsity_control, d, c-cfirst) ;
        end
        clast = c ;
    else
        fprintf ('.') ;
    end

end

fprintf ('\ntest234: all tests passed\n') ;

