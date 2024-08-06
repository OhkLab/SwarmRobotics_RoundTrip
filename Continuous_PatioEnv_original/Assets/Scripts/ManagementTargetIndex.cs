using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ManagementTargetIndex : MonoBehaviour
{
    public int numRobots;
    [NonSerialized] public int[] targetIndexs;
    
    void Awake()
    {
        targetIndexs = GenerateRandomArray(numRobots);
    }

    int[] GenerateRandomArray(int length)
    {
        if (length % 2 != 0)
        {
            throw new ArgumentException("配列の長さは偶数でなければなりません。");
        }

        int[] array = new int[length];
        int halfLength = length / 2;

        // 配列の半分に0、残り半分に1を追加
        for (int i = 0; i < halfLength; i++)
        {
            array[i] = 0;
        }
        for (int i = halfLength; i < length; i++)
        {
            array[i] = 1;
        }

        // 配列をランダムにシャッフル
        System.Random random = new System.Random();
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
        return array;
    }
}
