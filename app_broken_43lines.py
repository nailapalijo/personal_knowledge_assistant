
def improve_search_query(question):

    q = question.lower().strip()

    # 1. University questions FIRST
    if any(word in q for word in [
        "university", "college", "alma mater"
    ]):
        return (
            "Naila Palijo university name institution "
            "Mehran University of Engineering and Technology "
            "MUET Jamshoro graduated studied"
        )

    # 2. Education and degree questions
    elif any(word in q for word in [
        "degree", "education", "educational",
        "qualification", "graduate", "graduated"
    ]):
        return (
            "Naila Palijo Bachelor of Engineering "
            "Computer Systems Engineering MUET Jamshoro "
            "education qualification degree"
        )

    # 3. Personal name questions
    elif any(phrase in q for phrase in [
        "who am i", "what is my name",
        "my full name", "tell me my name"
    ]):
        return "Naila Palijo full name personal profile"

    # 4. Skills questions
    elif any(word in q for word in [
        "skill", "technical", "programming language"
    ]):
        return (
            "Naila Palijo technical skills Python Django Java "
            "JavaScript React HTML CSS Git"
        )

    return question
