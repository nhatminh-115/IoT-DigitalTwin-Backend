using System;
using TMPro;
using UnityEngine;

namespace DigitalTwin.Inference
{
    public class UnityInferenceUIBinder : MonoBehaviour
    {
        [SerializeField] private UnityInferenceClient inferenceClient;

        [Header("Feature Text Fields")]
        [SerializeField] private TMP_Text tempText;
        [SerializeField] private TMP_Text humidText;
        [SerializeField] private TMP_Text co2Text;
        [SerializeField] private TMP_Text tvocText;

        [Header("Metadata")]
        [SerializeField] private TMP_Text moduleText;
        [SerializeField] private TMP_Text issueTimeText;
        [SerializeField] private TMP_Text targetTimeText;
        [SerializeField] private TMP_Text statusText;

        private void Reset()
        {
            inferenceClient = FindObjectOfType<UnityInferenceClient>();
        }

        private void OnEnable()
        {
            if (inferenceClient == null)
            {
                return;
            }

            inferenceClient.OnPredictionUpdated += HandlePredictionUpdated;
            inferenceClient.OnRequestFailed += HandleRequestFailed;
        }

        private void OnDisable()
        {
            if (inferenceClient == null)
            {
                return;
            }

            inferenceClient.OnPredictionUpdated -= HandlePredictionUpdated;
            inferenceClient.OnRequestFailed -= HandleRequestFailed;
        }

        private void HandlePredictionUpdated(UnityPredictResponse response)
        {
            if (moduleText != null)
            {
                moduleText.text = $"Module: {response.module}";
            }

            if (issueTimeText != null)
            {
                issueTimeText.text = $"Issue: {response.forecast_issue_time}";
            }

            if (targetTimeText != null)
            {
                targetTimeText.text = $"Target: {response.forecast_target_time}";
            }

            UpdateFeatureText("Temp", tempText, response);
            UpdateFeatureText("Humid", humidText, response);
            UpdateFeatureText("CO2", co2Text, response);
            UpdateFeatureText("TVOC", tvocText, response);

            if (statusText != null)
            {
                statusText.text = "Inference status: Updated";
            }
        }

        private void HandleRequestFailed(string message)
        {
            if (statusText != null)
            {
                statusText.text = $"Inference status: {message}";
            }
        }

        private static void UpdateFeatureText(string featureName, TMP_Text targetText, UnityPredictResponse response)
        {
            if (targetText == null || response == null || response.features == null)
            {
                return;
            }

            FeaturePrediction match = null;
            for (int index = 0; index < response.features.Count; index++)
            {
                FeaturePrediction candidate = response.features[index];
                if (string.Equals(candidate.name, featureName, StringComparison.OrdinalIgnoreCase))
                {
                    match = candidate;
                    break;
                }
            }

            if (match == null)
            {
                targetText.text = $"{featureName}: N/A";
                return;
            }

            targetText.text =
                $"{featureName}\\nCurrent: {match.current:F3}\\nPredicted: {match.predicted:F3}\\nDelta: {match.delta:+0.000;-0.000;0.000}";
        }
    }
}
