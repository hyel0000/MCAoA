MCAoAN : An Improved Attention for Visual Question Answering 

이 레포지토리는 Yuhao Cui의 [MCAN](https://github.com/cuiyuhao1996/mcan-vqa) 구현을 참고하여, 
수정된 Attention Box인 SAoA, GAoA와 Multi-modal Attention Fusion 방식을 제안한 논문 An Improved Attention for Visual Question Answering을 구현한 **MCAoAN** 모델입니다.

---

## 🚀 프로젝트 개요

본 프로젝트는 **MCAN(Deep Modular Co-Attention Networks)** 모델을 기반으로 하였으며,  
Visual Question Answering(VQA) 성능 향상을 위해 다음과 같은 변경을 적용했습니다:

- 수정된 attention box인 Self Attention on Attention Box (SAoA), Guided Attention on Attention block (GAoA)
- 논문에서 제안한 multimodal attention fusion방식을 구현


---

## 📦 설치 및 실행

본 프로젝트는 MCAN 원본 코드 구조를 유지하고 있으며, 설정 및 실행 방법은 동일합니다.  
자세한 설명은 원작자 레포지토리를 참고하세요:

👉 https://github.com/cuiyuhao1996/mcan-vqa

---

## 📜 Citation

이 연구를 사용하신다면 다음 논문을 인용해주시길 바랍니다:

```bibtex
@article{cui2019mcan,
  title={Deep Modular Co-Attention Networks for Visual Question Answering},
  author={Cui, Yuhao et al.},
  journal={CVPR},
  year={2019},
  note={Original code base used as reference}
}

@misc{hyelee2025mcaoan,
  title={Implementation of MCAoAN: Object-aware Attention Extension of Modular Co-Attention Networks},
  author={Hye Lee Kim},
  year={2025},
  note={Implementation based on the paper "An Improved Attention for Visual Question Answering"}
}

