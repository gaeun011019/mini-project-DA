# Raw data

원본 Batch 파일은 각각 약 2GB 이상이므로 GitHub에 포함하지 않는다. 다음 명령으로 Batch1과 Batch2를 내려받는다.

```bash
python data/download_data.py
```

다운로드가 끝나면 다음 구조가 된다.

```text
data/
├── Batch1
├── Batch2
├── README.md
└── download_data.py
```

출처: [Stanford Chueh Group – Severson et al. Early Cycle Life Prediction dataset](https://chuehlab.stanford.edu/datasets)

다운로드 도중 연결이 끊기면 `.part` 파일이 남는다. 다시 시작하려면 해당 `.part` 파일을 삭제한 뒤 명령을 다시 실행한다.
