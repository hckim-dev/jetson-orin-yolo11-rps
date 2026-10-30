# 자격 증명과 저장소 검사

[README로 돌아가기](../README.md)

실제 비밀번호, 액세스 토큰, 개인 키는 커밋하지 않습니다. 필요한 경우 환경 변수나 `.gitignore`에 포함된 로컬 설정을 사용하고, 예제 파일에는 실제 값을 넣지 않습니다.

이 프로젝트의 [Secret scan 워크플로](../.github/workflows/secret-scan.yml)는 Gitleaks 8.30.1을 SHA-256 확인 후 설치하고, 추적 파일 스냅샷과 변경 커밋을 검사합니다. 규칙은 [.gitleaks.toml](../.gitleaks.toml)에 있습니다. 이 CI는 추론·게임 동작 테스트와 별개입니다.

로컬에 같은 버전의 Gitleaks를 준비한 뒤 실행합니다.

```bash
gitleaks git . --pre-commit --staged --config .gitleaks.toml --redact=100
```

전체 이력을 별도로 검사할 때는 다음 명령을 사용합니다.

```bash
gitleaks git . --log-opts="--all --full-history -m" --config .gitleaks.toml --redact=100
```

CI는 기준 커밋이 있으면 해당 커밋 이후 변경을 검사하고, 없으면 현재 트리와 최신 커밋을 검사합니다. 전체 이력 검사를 대신하지는 않습니다. 실제 자격 증명이 노출되었다면 발급처에서 폐기·교체해야 하며, 파일 삭제나 `.gitignore` 추가만으로 Git 이력에서 사라지지 않습니다. 이력 재작성은 별도 합의 없이 수행하지 않습니다.
