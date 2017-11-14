// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;

public class NavCamera : MonoBehaviour
{
    public bool IsGenerateImages = false;
    public bool ManualSnapshot = false;
    public bool SnapshotFile = false;
    public string SaveFilepath = @"S:\TMP\ImageGenerator";    
    public float TurnRate = 5f;
    public float TurnRateRandomRange = 5f;
    public float PitchRandomRange = 5f;
    public TextMesh StatusText;
    public float DelayTimer = 0.5f;
    public int MaxCars = 10;

    GameObject[] CarList;
    List<Vector3> CarPos = new List<Vector3>();

    int carIndex = 0;
    int carCount = 0;

    Vector3 targetPos;
    float Yaw;
    float Pitch;
    float Distance;
    float targetTime;
    bool isCar;
    string filenamePrefix;
    int fileSequenceId;

    
    void Start()
    {
        CarList = GameObject.FindGameObjectsWithTag("Car");
        carCount = MaxCars < 0 ? CarList.Length : MaxCars;
        
        Debug.LogFormat("Cars: {0}", carCount);
        Yaw = 0f;
        Pitch = 5f;
        Distance = 4f;
        filenamePrefix = "Car";
        fileSequenceId = 0;
        
        if(carIndex < carCount)
        {
            isCar = true;            
            foreach(var c in CarList)
            {
                CarPos.Add(c.transform.position);
            }

            targetPos = CarPos[carIndex] + Vector3.up * 0.8f;
            targetTime = Time.time;
            Debug.Log($"{filenamePrefix}-{carIndex:D3}-{(int)Yaw:D3}");
        }
    }

    void Update()
    {        
        if(Input.GetKey(KeyCode.Escape))
        {
            Application.Quit();
        }

        if(Time.time < targetTime)
        {
            return;
        }

        if (ManualSnapshot)
        {
            if (Input.GetKeyDown(KeyCode.Space))
            {
                if (SnapshotFile)
                {
                    ScreenCapture.CaptureScreenshot($"{SaveFilepath}\\Snapshot.png");
                }
                else
                {
                    ScreenCapture.CaptureScreenshot($"{SaveFilepath}\\Snapshot-{fileSequenceId:D3}.png");
                    fileSequenceId++;
                }
            }
        }
        else if(carIndex < carCount)
        {
            if(Yaw < 360f)
            {
                Yaw += TurnRate + Random.value * TurnRateRandomRange;
                var camRotation = Quaternion.Euler(Pitch + Random.value * PitchRandomRange, Yaw, 0f);
                transform.position = targetPos + Random.insideUnitSphere * 0.5f + camRotation * Vector3.back * Distance;
                transform.rotation = camRotation;

                Debug.Log($"{SaveFilepath}\\{filenamePrefix}-{carIndex:D3}-{(int)Yaw:D3}");
                if (IsGenerateImages)
                {
                    ScreenCapture.CaptureScreenshot($"{SaveFilepath}\\{filenamePrefix}-{carIndex:D3}-{(int)Yaw:D3}.png");
                }

                targetTime = Time.time + DelayTimer;
            }
            else
            {
                carIndex++;
                Yaw = 0f;
                if (carIndex < carCount)
                {
                    targetPos = CarPos[carIndex] + Vector3.up * 0.8f;
                }
            }
        }
        else if(isCar)
        {
            isCar = false;
            filenamePrefix = "NoCar";
            carIndex = 0;
            Yaw = 0f;
            foreach (var c in CarList)
            {
                c.SetActive(false);
            }
        } else
        {
            //Application.Quit();
        }
    }
}
