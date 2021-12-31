
#define BUILD_NORMAL(n, l, r, o) \
    [numthreads(64, 1, 1)]\
    void n(uint3 id : SV_DispatchThreadID) {\
        if (id.x > (uint)count) {\
            return;\
        }\
        _##n(left_##l, right_##r, output_##o, id.x);\
    }   

#define BUILD_INPLACE_L(n, l, r) \
    [numthreads(64, 1, 1)]\
    void n##_InplaceL(uint3 id : SV_DispatchThreadID) {\
        if (id.x > (uint)count) {\
            return;\
        }\
        _##n(left_##l, right_##r, left_##l, id.x);\
    }
#define BUILD_INPLACE_R(n, l, r) \
    [numthreads(64, 1, 1)]\
    void n##_InplaceR(uint3 id : SV_DispatchThreadID) {\
        if (id.x > (uint)count) {\
            return;\
        }\
        _##n(left_##l, right_##r, right_##r, id.x);\
    }
#define BUILD_SELF(n, l, o) \
    [numthreads(64, 1, 1)]\
    void n##_Self(uint3 id : SV_DispatchThreadID) {\
        if (id.x > (uint)count) {\
            return;\
        }\
        _##n(left_##l, left_##l, output_##o, id.x);\
    }
#define BUILD_INPLACE_SELF(n, l) \
    [numthreads(64, 1, 1)]\
    void n##_SelfInplace(uint3 id : SV_DispatchThreadID) {\
        if (id.x > (uint)count) {\
            return;\
        }\
        _##n(left_##l, left_##l, left_##l, id.x);\
    }

#define BUILD_AAA(n, l, r, o) \
    BUILD_NORMAL(n, l, r, o)\
    BUILD_INPLACE_L(n, l, r)\
    BUILD_INPLACE_R(n, l, r)\
    BUILD_SELF(n, l, o)\
    BUILD_INPLACE_SELF(n, l) 

#define BUILD_AAB(n, l, r, o) \
    BUILD_NORMAL(n, l, r, o)\
    BUILD_SELF(n, l, o)\

RWStructuredBuffer<float> left_float;
RWStructuredBuffer<float> right_float;
RWStructuredBuffer<float> output_float;
RWStructuredBuffer<bool> output_bool;
int count;

int lrank;
int rrank;
int drank;
int stride;

int batchCountL;
int batchCountR;

int4 lshape[16];
int4 rshape[16];
int4 oshape[16];



uint GetLShape(uint ind) {
    return lshape[ind / 4][ind % 4];
}
uint GetRShape(uint ind) {
    return rshape[ind / 4][ind % 4];
}
uint GetOShape(uint ind) {
    return oshape[ind / 4][ind % 4];
}



uint2 GetIndices(uint x) {
    uint ind_raw = x;

    uint batch = ind_raw / stride; // which batch are we on
    uint batch_ind = ind_raw % stride; // index inside that batch

    uint lind = 0;
    uint rind = 0;
    uint remaining = batch;
    uint strideO = (uint)count / (uint)stride;
    uint strideL = batchCountL;
    uint strideR = batchCountR;

    for (int j = drank; j > 0; j--) {
        int ll = lrank - j;
        int rr = rrank - j;
        int dd = drank - j;

        uint lsize = ll >= 0 ? GetLShape(ll) : 1;
        uint rsize = rr >= 0 ? GetRShape(rr) : 1;
        uint dsize = GetOShape(dd);

        strideO /= dsize;
        strideL /= lsize;
        strideR /= rsize;

        if (stride == 0) {
            break;
        }

        uint ind = remaining / strideO; // value at this index
        remaining %= strideO;


        // if broadcast (ie. shape is 1), do not update ind
        if (lsize != 1) {
            lind += ind * strideL;
        }
        if (rsize != 1) {
            rind += ind * strideR;
        }
    }

    uint loffset = stride * lind + batch_ind;
    uint roffset = stride * rind + batch_ind;
    return uint2(loffset, roffset);
}