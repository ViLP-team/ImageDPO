import re


def get_score_from_VLM_judge(results):
    pattern = re.compile(r"Score: (\d+)", re.MULTILINE)
    match = pattern.search(results)
    if match:
        return int(match.group(1))
    return None


def generate_grounding_dino_instruction(results_instruction):
    count = 0
    pattern = re.compile(
        r"Item Number:\s+(\d+)\s+Removed Object Description:\s+(.+?)\s+New Object Description:\s+(.+?)\s+New Image Description:\s+(.+?)(?=Item Number:\s+\d+|$)",
        re.DOTALL,
    )
    # Parse the string
    parsed_results = []
    matches = pattern.findall(results_instruction)
    for match in matches:
        try:
            item_number, remove_object, new_object, new_prompts = match
            parsed_results.append(
                {
                    "Item Number": int(item_number),
                    "Removed Object Description": remove_object.strip(),
                    "New Object Description": new_object.strip(),
                    "New Image Description": new_prompts.strip(),
                }
            )
            count += 1
        except:
            continue
    if count < 1:
        count = 0
        pattern = re.compile(
            r"^Item Number:\s+(\d+)\s+Removed Object Description:\s+(.+?)\s+New Object Description:\s+(.+?)\s+New Image Description:\s+(.+?)(?=Item Number:\s+\d+|$)",
            re.DOTALL | re.MULTILINE,
        )
        # Parse the string
        parsed_results = []
        matches = pattern.findall(results_instruction)
        for match in matches:
            try:
                item_number, remove_object, new_object, new_prompts = match
                parsed_results.append(
                    {
                        "Item Number": int(item_number),
                        "Removed Object Description": remove_object.strip(),
                        "New Object Description": new_object.strip(),
                        "New Image Description": new_prompts.strip(),
                    }
                )
            except:
                continue
        count = len(parsed_results)
    if count < 1:
        count = 0
        pattern = re.compile(
            r"\d+\.\s*Item Number:\s*\[(\d+)\]\s*Removed Object Description:\s*(.*?)\s*New Object Description:\s*(.*?)\s*New Image Description:\s*(.*?)(?=\d+\.\s*Item Number:|\Z)",
            re.DOTALL,
        )
        # Parse the string
        parsed_results = []
        matches = pattern.findall(results_instruction)
        for match in matches:
            try:
                item_number, remove_object, new_object, new_prompts = match
                parsed_results.append(
                    {
                        "Item Number": int(item_number),
                        "Removed Object Description": remove_object.strip(),
                        "New Object Description": new_object.strip(),
                        "New Image Description": new_prompts.strip(),
                    }
                )
            except:
                continue
        count = len(parsed_results)
    if count < 1:
        count = 0
        pattern = re.compile(
            r"Item Number:\s*\[(\d+)\]\s*Removed Object Description:\s*(.*?)\s*New Object Description:\s*(.*?)\s*New Image Description:\s*(.*?)(?=Item Number: \[\d+\]|\Z)",
            re.DOTALL,
        )

        # Parse the string
        parsed_results = []
        matches = pattern.findall(results_instruction)
        for match in matches:
            try:
                item_number, remove_object, new_object, new_prompts = match
                parsed_results.append(
                    {
                        "Item Number": int(item_number),
                        "Removed Object Description": remove_object.strip(),
                        "New Object Description": new_object.strip(),
                        "New Image Description": new_prompts.strip(),
                    }
                )
            except:
                continue
        count = len(parsed_results)
    if count < 1:
        count = 0
        pattern = re.compile(
            r"Item Number:\s*\[([\d, ]+)\]\s*Removed Object Description:\s*(.*?)\s*New Object Description:\s*(.*?)\s*New Image Description:\s*(.*?)(?=Item Number: \[\d+.*?\]|\Z)",
            re.DOTALL,
        )

        # Parse the string
        parsed_results = []
        matches = pattern.findall(results_instruction)
        for match in matches:
            try:
                item_number, remove_object, new_object, new_prompts = match
                parsed_results.append(
                    {
                        "Item Number": int(item_number),
                        "Removed Object Description": remove_object.strip(),
                        "New Object Description": new_object.strip(),
                        "New Image Description": new_prompts.strip(),
                    }
                )
            except:
                continue
        count = len(parsed_results)
    if count < 1:
        count = 0
        pattern = re.compile(
            r"Item Number: (\d+)\s+Item Removed: (.+?)\s+Item Replaced: (.+?)\s+New Image Description: (.+?)(?=Item Number: \d+|$)",
            re.DOTALL,
        )
        # Parse the string
        parsed_results = []
        matches = pattern.findall(results_instruction)
        for match in matches:
            try:
                item_number, item_removed, item_replaced, new_image_description = match
                parsed_results.append(
                    {
                        "Item Number": int(item_number),
                        "Removed Object Description": item_removed.strip(),
                        "New Object Description": item_replaced.strip(),
                        "New Image Description": new_image_description.strip(),
                    }
                )
            except:
                continue
        count = len(parsed_results)
    if count < 1:
        count = 0
        pattern = re.compile(
            r"\d+\.\s*Item Number:\s*(\d+)\s*\d+\.\s*Removed Object Description:\s*(.*?)\s*\d+\.\s*New Object Description:\s*(.*?)\s*\d+\.\s*New Image Description:\s*(.*?)(?=\d+\.\s*Item Number:|\Z)",
            re.DOTALL,
        )

        # Parse the string
        parsed_results = []
        matches = pattern.findall(results_instruction)
        for match in matches:
            try:
                item_number, remove_object, new_object, new_prompts = match
                parsed_results.append(
                    {
                        "Item Number": int(item_number),
                        "Removed Object Description": remove_object.strip(),
                        "New Object Description": new_object.strip(),
                        "New Image Description": new_prompts.strip(),
                    }
                )
            except:
                continue
        count = len(parsed_results)
    if count < 1:
        count = 0
        pattern = re.compile(
            r"\d+\.\s*Item Number:\s*\[(\d+)\]\s*\d+\.\s*Removed Object Description:\s*(.*?)\s*\d+\.\s*New Object Description:\s*(.*?)\s*\d+\.\s*New Image Description:\s*(.*?)(?=\d+\.\s*Item Number:|\Z)",
            re.DOTALL,
        )

        # Parse the string
        parsed_results = []
        matches = pattern.findall(results_instruction)
        for match in matches:
            try:
                item_number, remove_object, new_object, new_prompts = match
                parsed_results.append(
                    {
                        "Item Number": int(item_number),
                        "Removed Object Description": remove_object.strip(),
                        "New Object Description": new_object.strip(),
                        "New Image Description": new_prompts.strip(),
                    }
                )
            except:
                continue
        count = len(parsed_results)
    if count < 1:
        pattern = re.compile(
            r"^\d+\.\s*Item Number:\s*(\d+)\s*Item Removed:\s*(.*?)\s*New Object Description:\s*(.*?)\s*New Image Description:\s*(.*?)(?=\n\d+\.\s*Item Number:|\Z)",
            re.DOTALL | re.MULTILINE,
        )

        # Parse the string
        parsed_results = []
        matches = pattern.findall(results_instruction)
        for match in matches:
            try:
                item_number, item_removed, new_object, new_prompts = match
                parsed_results.append(
                    {
                        "Item Number": int(item_number),
                        "Item Removed": item_removed.strip(),
                        "New Object Description": new_object.strip(),
                        "New Image Description": new_prompts.strip(),
                    }
                )
            except:
                continue
        count = len(parsed_results)
    if count < 1:
        parsed_results = []
        matches = pattern.findall(results_instruction)
        for match in matches:
            try:
                (
                    section_number,
                    item_number,
                    item_description,
                    remove_object,
                    new_object,
                    new_prompts,
                ) = match
                parsed_results.append(
                    {
                        "Section Number": int(section_number),
                        "Item Number": int(item_number),
                        "Item Description": item_description.strip(),
                        "Removed Object Description": remove_object.strip(),
                        "New Object Description": new_object.strip(),
                        "New Image Description": new_prompts.strip(),
                    }
                )
            except:
                continue
        count = len(parsed_results)
    return count, parsed_results


def generate_instruction(results_instruction):
    count = 0
    pattern = re.compile(
        r"Item Number:\s+(\d+)\s+Tool Used:\s+([\w\s]+)\s+Text Prompt for Processing:\s+(.+?)(?=Item Number:\s+\d+|$)",
        re.DOTALL,
    )

    # Parse the string
    parsed_results = []
    matches = pattern.findall(results_instruction)
    for match in matches:
        try:
            item_number, tool_used, prompt = match
            parsed_results.append(
                {
                    "Item Number": int(item_number),
                    "Tool Used": tool_used,
                    "Text Prompt for Processing": prompt.strip(),
                }
            )
            count += 1
        except:
            continue
    if count <= 1:
        count = 0
        pattern = re.compile(
            r"^\s*(\d+)\.\s*Tool Used:\s*(.*?)\s*Text Prompt for Processing:\s*(.*?)(?=\s*\d+\.|$)",
            re.DOTALL | re.MULTILINE,
        )

        # Parse the string
        parsed_results = []
        matches = pattern.findall(results_instruction)
        for match in matches:
            try:
                item_number, tool_used, prompt = match
                parsed_results.append(
                    {
                        "Item Number": int(item_number),
                        "Tool Used": tool_used.strip(),
                        "Text Prompt for Processing": prompt.strip(),
                    }
                )
                count += 1
            except ValueError:
                continue
    if count <= 1:
        count = 0
        pattern = re.compile(r"^(\d+)\.\s*(.+?):\s*(.+)$", re.MULTILINE)

        # Parse the string
        parsed_results = []
        matches = pattern.findall(results_instruction)
        parsed_results = [
            {
                "Item Number": int(num),
                "Tool Used": tool.strip(),
                "Text Prompt for Processing": prompt.strip(),
            }
            for num, tool, prompt in matches
        ]
        count = len(parsed_results)

    if count <= 1:
        count = 0
        pattern = re.compile(r"\[(\d+)\]\s*(.+?):\s*(.+)", re.MULTILINE)

        # Parse the string
        parsed_results = []
        matches = pattern.findall(results_instruction)
        parsed_results = [
            {
                "Item Number": int(num),
                "Tool Used": tool.strip(),
                "Text Prompt for Processing": prompt.strip(),
            }
            for num, tool, prompt in matches
        ]
        count = len(parsed_results)

    if count <= 1:
        count = 0
        pattern = re.compile(
            r"\[(\d+)\]\s*Tool Used:\s*(.*?)\s*Text Prompt for Processing:\s*(.*?)(?=\[\d+\]\s*Tool Used:|\Z)",
            re.DOTALL,
        )

        # Parse the string
        parsed_results = []
        matches = pattern.findall(results_instruction)
        parsed_results = [
            {
                "Item Number": int(num),
                "Tool Used": tool.strip(),
                "Text Prompt for Processing": prompt.strip(),
            }
            for num, tool, prompt in matches
        ]
        count = len(parsed_results)

    return count, parsed_results


def extract_QA(results_QA):
    count = 0
    QA_save_dict = []

    pattern_QA = re.compile(
        r"Item Number: (?P<Number>\d+)\nQuestion: (?P<Question>.+?)\nAnswer: (?P<Answer>.+?)(?=\n\nItem Number: |\Z)",
        re.DOTALL,
    )
    matches = pattern_QA.finditer(results_QA)
    for match in matches:
        try:
            question = match.group("Question").strip()
            answer = match.group("Answer").strip()
            QA_save_dict.append({"Question": question, "Answer": answer})
            count += 1
        except:
            continue

    if count <= 1:
        count = 0
        pattern_QA = re.compile(
            r"Question: (?P<Question>.+?)\nAnswer: (?P<Answer>.*?)(?=\n\n\d+\. Question: |\Z)",
            re.DOTALL,
        )
        matches = pattern_QA.finditer(results_QA)
        for match in matches:
            try:
                question = match.group("Question").strip()
                answer = match.group("Answer").strip()
                QA_save_dict.append({"Question": question, "Answer": answer})
                count += 1
            except:
                continue

    if count <= 1:
        count = 0
        pattern_QA = re.compile(
            r"(?<=\d\.\s)(?:What's|What|Where|How).+?\?\nAnswer: .+?(?=\n\d\. |\Z)",
            re.DOTALL,
        )
        matches = pattern_QA.finditer(results_QA)
        for match in matches:
            try:
                qa_pair = match.group(0).strip().split("\nAnswer: ")
                question = qa_pair[0].split("?")[0].strip() + "?"
                answer = qa_pair[1].strip()
                QA_save_dict.append({"Question": question, "Answer": answer})
                count += 1
            except:
                continue

    return count, QA_save_dict


def extract_justify_rating(results_QA):
    count = 0
    QA_save_dict = []

    pattern_QA = re.compile(
        r"Justify:\s*(?P<Justify>.+?)\nScore:\s*(?P<Score>\d+)\n",
        re.MULTILINE,
    )
    matches = pattern_QA.finditer(results_QA)
    for match in matches:
        try:
            answer = match.group("Justify").strip()
            score = match.group("Score").strip()
            QA_save_dict.append({"Justify": answer, "Score": score})
            count += 1
        except:
            print("Error processing match:")
            print("Original string:", results_QA)
            continue

    if count <= 0:
        pattern_QA = re.compile(
            r"Justify:\s*(?P<Justify>.*?)\nScore:\s*(?P<Score>\d+)\n", re.DOTALL
        )
        QA_save_dict = []

        # Find all matches
        matches = pattern_QA.finditer(results_QA)
        count = 0

        # Process each match
        for match in matches:
            try:
                answer = match.group("Justify").strip()
                score = match.group("Score").strip()
                QA_save_dict.append({"Justify": answer, "Score": score})
                count += 1
            except Exception as e:
                print("Error processing match:", e)
                print("Original string:", results_QA)
                continue
    if count <= 0:
        pattern_QA = re.compile(
            r"Justify:\s*(?P<Justify>.*?)\s*Score:\s*(?P<Score>\d+)", re.DOTALL
        )
        QA_save_dict = []

        # Find all matches
        matches = pattern_QA.finditer(results_QA)
        count = 0

        # Process each match
        for match in matches:
            try:
                answer = match.group("Justify").strip()
                score = match.group("Score").strip()
                QA_save_dict.append({"Justify": answer, "Score": score})
                count += 1
            except Exception as e:
                print("Error processing match:", e)
                print("Original string:", results_QA)
                continue
    if count <= 0:
        pattern_QA = re.compile(
            r"(?P<Justify>.+?)\s*Score:\s*(?P<Score>\d+)", re.DOTALL
        )
        # List to save the results
        QA_save_dict = []

        # Find all matches
        matches = pattern_QA.finditer(results_QA)
        count = 0

        # Process each match
        for match in matches:
            try:
                answer = match.group("Justify").strip()
                score = match.group("Score").strip()
                QA_save_dict.append({"Justify": answer, "Score": score})
                count += 1
            except Exception as e:
                print("Error processing match:", e)
                print("Original string:", results_QA)
                continue

    return count, QA_save_dict


def extract_answer_rating(results_QA):
    count = 0
    QA_save_dict = []

    pattern_QA = re.compile(
        r"Answer:\s*(?P<Answer>.+?)\nJustify:\s*(?P<Justify>.+?)\nScore:\s*(?P<Score>\d+)\n",
        re.MULTILINE | re.DOTALL,
    )
    matches = pattern_QA.finditer(results_QA)
    for match in matches:
        try:
            answer = match.group("Answer").strip()
            justify = match.group("Justify").strip()
            score = match.group("Score").strip()
            QA_save_dict.append({"Answer": answer, "Justify": justify, "Score": score})
            count += 1
        except Exception as e:
            print("Error processing match:", e)
            print("Original string:", results_QA)
            continue

    if count <= 0:
        pattern_QA = re.compile(
            r"Answer:\s*(?P<Answer>.*?)\nJustify:\s*(?P<Justify>.*?)\nScore:\s*(?P<Score>\d+)\n",
            re.DOTALL,
        )
        QA_save_dict = []

        # Find all matches
        matches = pattern_QA.finditer(results_QA)
        count = 0

        # Process each match
        for match in matches:
            try:
                answer = match.group("Answer").strip()
                justify = match.group("Justify").strip()
                score = match.group("Score").strip()
                QA_save_dict.append(
                    {"Answer": answer, "Justify": justify, "Score": score}
                )
                count += 1
            except Exception as e:
                print("Error processing match:", e)
                print("Original string:", results_QA)
                continue

    if count <= 0:
        pattern_QA = re.compile(
            r"Answer:\s*(?P<Answer>.*?)\s*Justify:\s*(?P<Justify>.*?)\s*Score:\s*(?P<Score>\d+)",
            re.DOTALL,
        )
        QA_save_dict = []

        # Find all matches
        matches = pattern_QA.finditer(results_QA)
        count = 0

        # Process each match
        for match in matches:
            try:
                answer = match.group("Answer").strip()
                justify = match.group("Justify").strip()
                score = match.group("Score").strip()
                QA_save_dict.append(
                    {"Answer": answer, "Justify": justify, "Score": score}
                )
                count += 1
            except Exception as e:
                print("Error processing match:", e)
                print("Original string:", results_QA)
                continue

    if count <= 0:
        pattern_QA = re.compile(
            r"(?P<Answer>.+?)\s*Justify:\s*(?P<Justify>.+?)\s*Score:\s*(?P<Score>\d+)",
            re.DOTALL,
        )
        QA_save_dict = []

        # Find all matches
        matches = pattern_QA.finditer(results_QA)
        count = 0

        # Process each match
        for match in matches:
            try:
                answer = match.group("Answer").strip()
                justify = match.group("Justify").strip()
                score = match.group("Score").strip()
                QA_save_dict.append(
                    {"Answer": answer, "Justify": justify, "Score": score}
                )
                count += 1
            except Exception as e:
                print("Error processing match:", e)
                print("Original string:", results_QA)
                continue
    if count <= 0:
        pattern_QA = re.compile(
            r"Answer:\s*\[(?P<Answer>.+?)\]\s*Justify:\s*\[(?P<Justify>.+?)\]\s*Score:\s*\[(?P<Score>\d+)[^\]]*\]",
            re.DOTALL,
        )
        # List to save the results
        QA_save_dict = []

        # Find all matches
        matches = pattern_QA.finditer(results_QA)
        count = 0

        # Process each match
        for match in matches:
            try:
                answer = match.group("Answer").strip()
                justify = match.group("Justify").strip()
                score = match.group("Score").strip()
                QA_save_dict.append(
                    {"Answer": answer, "Justify": justify, "Score": score}
                )
                count += 1
            except Exception as e:
                print("Error processing match:", e)
                print("Original string:", results_QA)
                continue
    if count <= 0:
        pattern_QA = re.compile(
            r"Answer:\s*(?P<Answer>.+?)\nJustification:\s*(?P<Justification>(?:\*.*?\n)+)Score:\s*(?P<Score>\d+)",
            re.DOTALL,
        )
        # List to save the results
        QA_save_dict = []

        # Find all matches
        matches = pattern_QA.finditer(results_QA)
        count = 0

        # Process each match
        for match in matches:
            try:
                answer = match.group("Answer").strip()
                justify = match.group("Justify").strip()
                score = match.group("Score").strip()
                QA_save_dict.append(
                    {"Answer": answer, "Justify": justify, "Score": score}
                )
                count += 1
            except Exception as e:
                print("Error processing match:", e)
                print("Original string:", results_QA)
                continue

    return count, QA_save_dict
