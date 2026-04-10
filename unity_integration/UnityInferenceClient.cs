using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

namespace DigitalTwin.Inference
{
    [Serializable]
    public class FeaturePrediction
    {
        public string name;
        public float current;
        public float predicted;
        public float delta;
    }

    [Serializable]
    public class UnityPredictResponse
    {
        public string module;
        public string forecast_issue_time;
        public string forecast_target_time;
        public List<FeaturePrediction> features;
    }

    public class UnityInferenceClient : MonoBehaviour
    {
        [Header("API")]
        [SerializeField] private string apiBaseUrl = "http://127.0.0.1:8000";
        [SerializeField] private string moduleCode = "M1";

        [Header("Polling")]
        [SerializeField] private bool fetchOnStart = true;
        [SerializeField] private bool autoPoll = true;
        [SerializeField] private float pollIntervalSeconds = 180f;

        public UnityPredictResponse LatestResponse { get; private set; }
        public string LastError { get; private set; }

        public event Action<UnityPredictResponse> OnPredictionUpdated;
        public event Action<string> OnRequestFailed;

        private Coroutine pollingRoutine;

        private void Start()
        {
            if (fetchOnStart)
            {
                RequestPredictionNow();
            }

            if (autoPoll)
            {
                pollingRoutine = StartCoroutine(PollingLoop());
            }
        }

        private void OnDisable()
        {
            if (pollingRoutine != null)
            {
                StopCoroutine(pollingRoutine);
                pollingRoutine = null;
            }
        }

        public void RequestPredictionNow()
        {
            StartCoroutine(RequestPredictionCoroutine());
        }

        public void SetModule(string newModule)
        {
            if (string.IsNullOrWhiteSpace(newModule))
            {
                return;
            }

            moduleCode = newModule.Trim();
            RequestPredictionNow();
        }

        public bool TryGetFeature(string featureName, out FeaturePrediction featurePrediction)
        {
            featurePrediction = null;
            if (LatestResponse == null || LatestResponse.features == null)
            {
                return false;
            }

            for (int index = 0; index < LatestResponse.features.Count; index++)
            {
                FeaturePrediction candidate = LatestResponse.features[index];
                if (string.Equals(candidate.name, featureName, StringComparison.OrdinalIgnoreCase))
                {
                    featurePrediction = candidate;
                    return true;
                }
            }

            return false;
        }

        private IEnumerator PollingLoop()
        {
            while (true)
            {
                yield return RequestPredictionCoroutine();
                yield return new WaitForSeconds(pollIntervalSeconds);
            }
        }

        private IEnumerator RequestPredictionCoroutine()
        {
            string url = BuildPredictUrl();
            using UnityWebRequest request = UnityWebRequest.Get(url);
            request.timeout = 20;

            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                LastError = $"HTTP request failed: {request.error}";
                OnRequestFailed?.Invoke(LastError);
                yield break;
            }

            string payload = request.downloadHandler.text;
            UnityPredictResponse response;
            try
            {
                response = JsonUtility.FromJson<UnityPredictResponse>(payload);
            }
            catch (Exception exception)
            {
                LastError = $"JSON parse error: {exception.Message}";
                OnRequestFailed?.Invoke(LastError);
                yield break;
            }

            if (response == null || response.features == null)
            {
                LastError = "Invalid API response: missing features array.";
                OnRequestFailed?.Invoke(LastError);
                yield break;
            }

            LatestResponse = response;
            LastError = null;
            OnPredictionUpdated?.Invoke(response);
        }

        private string BuildPredictUrl()
        {
            string baseUrl = apiBaseUrl.TrimEnd('/');
            string escapedModule = UnityWebRequest.EscapeURL(moduleCode);
            return $"{baseUrl}/unity/predict?module={escapedModule}";
        }
    }
}
