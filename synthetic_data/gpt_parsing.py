import re

def parse_gpt_field_lists(field_names, gpt_output, throw_exception_on_failure=False):
    output_dict = {}
    overall_pattern = r'<response>\n'
    # TODO: double check this backslash on the hyphen doesn't break anything
    subject_pattern = r"[A-Z|a-z|2|3| |&|/|\(|\)|,|'|\-]+"
    degree_type_pattern = r'[A-Z|a-z| |\.|\(|\)]+'
    for field in field_names:
        overall_pattern += fr'{field}: \[((?:\("{subject_pattern}", "{degree_type_pattern}"\)(?:, )?)+)\]\n'
    overall_pattern += r'</response>'
    overall_match = re.search(overall_pattern, gpt_output)
    if overall_match:
        #output format is correct
        assert(len(overall_match.groups()) == len(field_names))
        for field_name, field_list_string in zip(field_names, overall_match.groups()):
            pairs_list = re.findall(fr'\("({subject_pattern})", "({degree_type_pattern})"\)', field_list_string)
            output_dict[field_name] = pairs_list
    elif throw_exception_on_failure:
        raise Exception('Failed to parse gpt output')

    return output_dict

def parse_gpt_field_lists_multi(field_names, gpt_output, expected_response_num, throw_exception_on_failure=False):
    output_dicts = []
    overall_pattern = r'<response>\n'
    # TODO: double check this backslash on the hyphen doesn't break anything
    subject_pattern = r"[A-Z|a-z|2|3| |&|/|\(|\)|,|'|\-]+"
    degree_type_pattern = r'[A-Z|a-z| |\.|\(|\)]+'

    categories_pattern_capturing = ''
    for field in field_names:
        categories_pattern_capturing += fr'{field}: \[((?:\("{subject_pattern}", "{degree_type_pattern}"\)(?:, )?)+)\]\n'

    categories_pattern_non_capturing = ''
    for field in field_names:
        categories_pattern_non_capturing += fr'{field}: \[(?:\("{subject_pattern}", "{degree_type_pattern}"\)(?:, )?)+\]\n'

    ignore_pattern_capturing = r'(Education level not appropriate for query job)\n'
    ignore_pattern_non_capturing = r'(?:Education level not appropriate for query job)\n'

    overall_pattern += fr'(?:{categories_pattern_non_capturing}|{ignore_pattern_non_capturing})'
    overall_pattern += r'</response>'
    response_matches = re.findall(overall_pattern, gpt_output)
    if len(response_matches) == expected_response_num:
        for response_match in response_matches:

            ignore_case_pattern = fr'<response>\n{ignore_pattern_capturing}</response>'
            if re.match(ignore_case_pattern, response_match):
                output_dicts.append(None)
                continue

            categories_case_pattern = fr'<response>\n{categories_pattern_capturing}</response>'
            categories_match = re.search(categories_case_pattern, response_match)

            if (not categories_match) or (len(categories_match.groups()) != len(field_names)):
                if throw_exception_on_failure:
                    raise Exception('Failed to parse gpt output')
                else:
                    return []
            assert(categories_match)
            assert len(categories_match.groups()) == len(field_names)
            output_dict = {}
            for field_name, field_list_string in zip(field_names, categories_match.groups()):
                pairs_list = re.findall(fr'\("({subject_pattern})", "({degree_type_pattern})"\)', field_list_string)
                output_dict[field_name] = pairs_list

            output_dicts.append(output_dict)

    elif throw_exception_on_failure:
        raise Exception('Failed to parse gpt output')

    return output_dicts