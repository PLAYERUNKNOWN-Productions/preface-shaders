// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

float elu(in float p_a)
{
    return ((p_a > 0) ? p_a : exp(p_a) - 1.0f);
}

void elu_func(uint3 p_dispatch_thread_id)
{
    
}
