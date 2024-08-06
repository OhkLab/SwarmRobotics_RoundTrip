using System;
using UnityEngine;
using UnityEngine.UI; //add (22/9/7)
using UnityStandardAssets.CrossPlatformInput;

[RequireComponent(typeof (Image))]
//[RequireComponent(typeof (GUITexture))]
//[RequireComponent(typeof (UI.Image))]
public class ForcedReset : MonoBehaviour
{
    private void Update()
    {
        // if we have forced a reset ...
        if (CrossPlatformInputManager.GetButtonDown("ResetObject"))
        {
            //... reload the scene
            Application.LoadLevelAsync(Application.loadedLevelName);
        }
    }
}
