class QwenUncensoredStorySplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "auto_split": ("BOOLEAN", {"default": True}),
                "custom_delimiter": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt_1", "prompt_2", "prompt_3", "prompt_4", "debug_info")
    FUNCTION = "split_story"
    CATEGORY = "Qwen Uncensored/Tools"

    def split_story(self, text, auto_split=True, custom_delimiter=""):
        if not text:
            return "", "", "", "", "Empty input"

        text = text.strip()
        if auto_split and not custom_delimiter:
            delimiter = "\n\n\n" if "\n\n\n" in text else "\n\n" if "\n\n" in text else "\n"
        else:
            delimiter = custom_delimiter or "\n\n"

        parts = text.split(delimiter)
        prompts = [
            part.strip()
            for part in parts
            if part.strip()
            and not part.strip().startswith("Prompt")
            and "content describing" not in part.strip()
        ]
        found_count = len(prompts)
        prompts = (prompts + ["", "", "", ""])[:4]
        debug = f"Found {len(parts)} parts, cleaned to {found_count} prompts"
        return prompts[0], prompts[1], prompts[2], prompts[3], debug


NODE_CLASS_MAPPINGS = {"QwenUncensoredStorySplit": QwenUncensoredStorySplit}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenUncensoredStorySplit": "Qwen Uncensored - Story Split"}
