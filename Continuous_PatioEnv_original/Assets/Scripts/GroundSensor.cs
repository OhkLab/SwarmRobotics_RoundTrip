using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GroundSensor : MonoBehaviour
{
    public GameObject Manager;
    public int robotIndex;
    [NonSerialized] public int cptCount;
    private int _firstLandmarkNumber;

    private void Start()
    {
        int[] targetIndexes = Manager.GetComponent<ManagementTargetIndex>().targetIndexs;
        _firstLandmarkNumber = targetIndexes[robotIndex];
    }

    public void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Landmark1"))
        {
            if (_firstLandmarkNumber == 0)
            {
                if (cptCount % 2 == 0)
                {
                    cptCount++;
                }
            }
            else
            {
                if (cptCount % 2 == 1)
                {
                    cptCount++;
                }
            }
        }
        
        if (other.CompareTag("Landmark2"))
        {
            if (_firstLandmarkNumber == 1)
            {
                if (cptCount % 2 == 0)
                {
                    cptCount++;
                }
            }
            else
            {
                if (cptCount % 2 == 1)
                {
                    cptCount++;
                }
            }
        }
    }
}
