
def calculate_CCR(recommendations, similarity_scores, completed_connections):
    total_score = 0
    total_recommendations = 0

    for user, recs in recommendations.items():
        for recommended_user in recs:
            total_score += similarity_scores[(user, recommended_user)] * completed_connections[(user, recommended_user)]
            total_recommendations += 1

    return total_score / total_recommendations if total_recommendations > 0 else 0
