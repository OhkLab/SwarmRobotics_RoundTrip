using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

public class ScreenShotCapture : MonoBehaviour
{
    public string directory;
    [SerializeField] private Camera _camera;
    public int depth = 24;
    private int _count = 0;
    private float _isCapture;

    private void Start()
    {
        var envParams = Academy.Instance.EnvironmentParameters;
        _isCapture = envParams.GetWithDefault("is_capture", 0.0f);

        if (Mathf.Approximately(_isCapture, 1.0f))
        {
            string dirPath = Application.persistentDataPath + "/" + directory;
            if (!Directory.Exists(dirPath))
            {
                Directory.CreateDirectory(dirPath);
            }
        }
    }

    private void Update()
    {
        if (Mathf.Approximately(_isCapture, 1.0f))
        {
            name = Application.persistentDataPath + "/" + directory + "/img_" + _count.ToString() + ".png";
            // スクリーンショットを保存
            CaptureScreenShot(name);
        
            // メモリ解放
            if (_count % 500 == 0)
            {
                System.GC.Collect();
                Resources.UnloadUnusedAssets();
            }
            _count++;
        }
    }

    private void CaptureScreenShot(string filePath)
    {
        var rt = new RenderTexture(_camera.pixelWidth, _camera.pixelHeight, depth);
        var prev = _camera.targetTexture;
        _camera.targetTexture = rt;
        _camera.Render();
        _camera.targetTexture = prev;
        RenderTexture.active = rt;

        var screenShot = new Texture2D(
            _camera.pixelWidth,
            _camera.pixelHeight,
            TextureFormat.RGB24,
            false
        );
        screenShot.ReadPixels(new Rect(0, 0, screenShot.width, screenShot.height), 0, 0);
        screenShot.Apply();

        var bytes = screenShot.EncodeToPNG();
        Destroy(screenShot);
        System.IO.File.WriteAllBytes(filePath, bytes);
    }
}
