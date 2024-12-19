// Copyright (c) PLAYERUNKNOWN Productions. All Rights Reserved.

float prelu(in float p_x, in float p_alpha)
{
    return p_x < 0.0f ? p_alpha * p_x : p_x;
}
