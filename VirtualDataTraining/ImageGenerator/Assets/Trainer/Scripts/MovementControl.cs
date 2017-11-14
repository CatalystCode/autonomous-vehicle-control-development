// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovementControl : MonoBehaviour
{
    public float Speed = 6.0F;
    public float TurnRate = 90.0F;

    private Vector3 moveDirection = Vector3.zero;

    void Update()
    {
        CharacterController controller = GetComponent<CharacterController>();
    
        moveDirection = new Vector3(Input.GetAxis("Horizontal"), 0, Input.GetAxis("Vertical"));
        moveDirection = transform.TransformDirection(moveDirection);        
        controller.transform.Rotate(new Vector3(0f, Input.GetAxis("HorizontalTurn"), 0f) * TurnRate * Time.deltaTime);
        controller.Move(moveDirection * Speed * Time.deltaTime);
    }
}
