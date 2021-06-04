from math import floor
## 저장 모듈
import os
import argparse

## 데이터 처리용 모듈
import numpy as np
import pandas as pd


def train(symbol, inter_train, valid_iter, iter_test, data_names, label_names, save_dir):
    # 학습 정보와 결과 저장
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    printFile = open(os.path.join(args.save_dir, 'log.txt'), 'w')
    def print_to_file(msg):
        print(msg)
        print(msg, file = printFile, flush = True) #flush인수가True로 설정된 경우 print()함수는 효율성을 높이기 위해 데이터를 버퍼링하지 않고 각 호출에서 계속 강제적으로 플러시
    print_to_file('Epoch    Traning Cor     Validation Cor')

    ## 이전 에폭에 대한 값을 저장하여 성능 향상의 한계점 설정
    ## 진행이 느릴 경우 일찍 끝내도록 해줌
    buf = RingBuffer(THRESHOLD_EPOCHS) #링버퍼는 미리 정해진 크기까지만 데이터를 저장할 수 있음
    old_val = 0

    ## 학습 과정
    for epoch in range(args.n_epochs) : 
        inter_train.reset()
        iter_val.reset()
        for batch in inter_train:
            # 예측값 계산
            module.forward(batch, is_train = True)
            # 경삿값 계산
            module.backward()
            # 파라미터 갱신
            module.update()
        
        ## 학습 데이터셋의 결과
        train_pred = module.predic(inter_train).asnumpy()
        train_label = iter_train.label[0][1].asnumpy()
        train_perf = perf.write_eval(train_pred, train_label, save_dir, 'train', epoch)

        ## 검증 데이터셋의 결과
        val_pred = module.predic(inter_val).asnumpy()
        val_label = iter_val.label[0][1].asnumpy()
        val_perf = perf.write_eval(val_pred, val_label, save_dir, 'valid', epoch)

        ## 테스트 데이터셋의 결과
        test_pred = module.predic(inter_test).asnumpy()
        test_label = iter_test.label[0][1].asnumpy()
        test_perf = perf.write_eval(test_pred, test_label, save_dir, 'tst', epoch)
        print_to_file('TESTING PERFORMANCE')
        print_to_file(test_perf)

        if epoch>0: 
            buf.append(val_perf['COR']- old_val)
        if epoch > 2:
            vals = buf.get()
            vals = [v for v in vals if v!=0]
            if sum([v<COR_THRESHOLD for v in vals]) == len(vals) :
                print_to_file('EARLY EXIT')
                break
        old_val = val_perf['COR']

def evaluate_and_write(pred, label, save_dir, mode, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    pred_df = pd.DataFrame(pred)
    label_df =pd.DataFrame(label)
    pred_df.to_csv(os.path.join(save_dir, '%s_pred%d.csv'%(mode, epoch)))
    label_df.to_csv(os.path.join(save_dir, '%s_label%d.csv'%(mode, epoch)))

    return {'COR': COR(label,pred)}

## correlation
def COR(label, pred):
    label_demeaned = label - label.mean(0)
    label_sumsquares = np.sum(np.square(label_demeaned),0)

    pred_demeaned = pred - pred.mean(0)
    pred_sumsquares = np.sum(np.square(pred_demeaned),0)

    cor_coef = np.diagonal(np.dot(label_demeaned.T), pred_demeaned) / np.sqrt(label_sumsquares* pred_sumsquares)
    return np.nanmean(cor_coef)

if __name__ == '__main__':
    # 커맨드라인에서 입력받는 인수를 파싱
    args = parser.parse_args()

    # 데이터 반복자 생성
    iter_train, iter_val, iter_test = prepare_iters(args.data_dir, args.win, args.h, args.model, args.batch_n)

    # 기호 준비
    input_feature_shape = iter_train.provide_data[0][1]
    X =
    Y =

    train(symbol, inter_train, valid_iter, iter_test, data_names, label_names, save_dir)







