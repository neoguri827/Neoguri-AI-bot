    def _discover_models(self) -> List[str]:
        try:
            raw_models = list(self.client.models.list())
        except Exception as e:
            logger.warning(f"모델 목록 조회 실패, 기본 후보 목록 사용: {e}")
            return []

        EXCLUDE_KEYWORDS = [
            "tts", "audio", "image", "vision", "embedding",
            "aqa", "live", "veo", "imagen", "learnlm",
        ]

        usable = []
        for m in raw_models:
            name = getattr(m, "name", None)
            if not name:
                continue
            short_name = name.split("/")[-1]
            if any(k in short_name.lower() for k in EXCLUDE_KEYWORDS):
                continue
            supported = (
                getattr(m, "supported_actions", None)
                or getattr(m, "supported_generation_methods", None)
                or []
            )
            if supported and not any("generatecontent" in str(s).lower() for s in supported):
                continue
            usable.append(short_name)

        if not usable:
            return []

        latest_flash = [c for c in usable if "latest" in c.lower() and "flash" in c.lower()]
        other_flash = [c for c in usable if "flash" in c.lower() and c not in latest_flash]
        others = [c for c in usable if c not in latest_flash and c not in other_flash]
        ordered = latest_flash + other_flash + others
        logger.info(f"실시간 조회된 사용 가능 모델(우선순위 정렬): {ordered[:6]}{'...' if len(ordered) > 6 else ''}")
        return ordered
