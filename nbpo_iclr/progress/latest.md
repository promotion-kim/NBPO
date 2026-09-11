[KST 07:43] 본문 완료: objective 5/9, cross-play 0/4, DPO weights 0/7, capability 7/36 (lm-eval 21)
진행/다음: 큐 BLOCKED 1, DONE 178, FAILED 6, PENDING 55, READY 14, RUNNING 1 → uf4_crossplay_judge_base__vs__fixedref_mse_s42_t2048 (prio 64)
GPU0: uf4_train_maxmin_mse_s43, 0%, 32385/143771 MiB | GPU1: uf4_train_maxmin_mse_s43, 0%, 1497/143771 MiB
GPU2: uf4_train_maxmin_mse_s43, 0%, 1497/143771 MiB | GPU3: uf4_train_maxmin_mse_s43, 0%, 1497/143771 MiB
ETA: 핵심 기전 [1.5-2.2h] / 본문 첫 전체 평가 [산정 대기: mopo_adapt/prosper_adapt 충실 구현 미완] / 본문 최종 [산정 대기: PROSPER/MOPO 구현 + cross-play bank 미확정]
이번 완료: 새 완료 결과 없음
PDF: 성공 07:43, 46쪽, nbpo_iclr/main_v6.pdf
Blocker/복구: 표 자동기입 실패:  296, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/sjkim/MNPO/nbpo_iclr/progress/fill_exhibits.py", line 274, in main
    "results": {a: {c: report["results"][a][c] for c in CRITERIA} for a in arms},
                       ~~~~~~~~~~~~~~~~~^^^
KeyError: 'devsel_maxmin_mse_s42'

