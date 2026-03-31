#if UNITY_EDITOR && !COMPILER_UDONSHARP
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using GaussianSplatting.Runtime;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;

namespace GaussianSplatting.Editor.Utils
{
    // BANTAD VERSION: Tog bort sh1-shF för att spara 75% RAM
    public struct InputSplatData
    {
        public Vector3 pos;
        public Vector3 nor;
        public Vector3 dc0;
        public float opacity;
        public Vector3 scale;
        public Quaternion rot;
    }

    [BurstCompile]
    public class GaussianFileReader
    {
        public static int ReadFileHeader(string filePath)
        {
            int vertexCount = 0;
            if (File.Exists(filePath))
            {
                if (isPLY(filePath))
                    PLYFileReader.ReadFileHeader(filePath, out vertexCount, out _, out _);
                else if (isSPZ(filePath))
                    SPZFileReader.ReadFileHeader(filePath, out vertexCount);
            }
            return vertexCount;
        }

        public static unsafe void ReadFile(string filePath, out NativeArray<InputSplatData> splats)
        {
            if (isPLY(filePath))
            {
                List<(string, PLYFileReader.ElementType)> attributes;
                PLYFileReader.ReadFile(filePath, out var splatCount, out var vertexStride, out attributes, out NativeArray<byte>[] plyRawDataChunks);
                string attrError = CheckPLYAttributes(attributes);
                if (!string.IsNullOrEmpty(attrError))
                    throw new IOException($"PLY file is probably not a Gaussian Splat file? Missing properties: {attrError}");
                
                splats = PLYDataToSplats(plyRawDataChunks, splatCount, vertexStride, attributes);
                
                foreach (var chunk in plyRawDataChunks)
                {
                    if (chunk.IsCreated)
                        chunk.Dispose();
                }

                // ReorderSHs behövs inte längre!
                LinearizeData(splats);
                return;
            }
            if (isSPZ(filePath))
            {
                SPZFileReader.ReadFile(filePath, out splats);
                return;
            }
            throw new IOException($"File {filePath} is not a supported format");
        }

        static bool isPLY(string filePath) => filePath.EndsWith(".ply", true, CultureInfo.InvariantCulture);
        static bool isSPZ(string filePath) => filePath.EndsWith(".spz", true, CultureInfo.InvariantCulture);

        static string CheckPLYAttributes(List<(string, PLYFileReader.ElementType)> attributes)
        {
            string[] required = { "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3" };
            List<string> missing = required.Where(req => !attributes.Contains((req, PLYFileReader.ElementType.Float))).ToList();
            if (missing.Count == 0)
                return null;
            return string.Join(",", missing);
        }

        static unsafe NativeArray<InputSplatData> PLYDataToSplats(NativeArray<byte>[] inputChunks, int count, int stride, List<(string, PLYFileReader.ElementType)> attributes)
        {
            NativeArray<int> fileAttrOffsets = new NativeArray<int>(attributes.Count, Allocator.Temp);
            int offset = 0;
            for (var ai = 0; ai < attributes.Count; ai++)
            {
                var attr = attributes[ai];
                fileAttrOffsets[ai] = offset;
                offset += PLYFileReader.TypeToSize(attr.Item2);
            }

            // BANTAD: Vi läser inte in någon "f_rest_" data alls.
            string[] splatAttributes =
            {
                "x", "y", "z", "nx", "ny", "nz",
                "f_dc_0", "f_dc_1", "f_dc_2",
                "opacity",
                "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3",                
            };
            
            Assert.AreEqual(UnsafeUtility.SizeOf<InputSplatData>() / 4, splatAttributes.Length);
            NativeArray<int> srcOffsets = new NativeArray<int>(splatAttributes.Length, Allocator.Temp);
            for (int ai = 0; ai < splatAttributes.Length; ai++)
            {
                int attrIndex = attributes.IndexOf((splatAttributes[ai], PLYFileReader.ElementType.Float));
                int attrOffset = attrIndex >= 0 ? fileAttrOffsets[attrIndex] : -1;
                srcOffsets[ai] = attrOffset;
            }
            
            NativeArray<InputSplatData> dst = new NativeArray<InputSplatData>(count, Allocator.Persistent);
            byte* dstPtr = (byte*)dst.GetUnsafePtr();
            int dstStride = UnsafeUtility.SizeOf<InputSplatData>();

            int currentSplatOffset = 0;
            for (int i = 0; i < inputChunks.Length; i++)
            {
                int splatsInThisChunk = inputChunks[i].Length / stride;
                ReorderPLYData(
                    splatsInThisChunk, 
                    (byte*)inputChunks[i].GetUnsafeReadOnlyPtr(), 
                    stride, 
                    dstPtr + ((long)currentSplatOffset * dstStride),
                    dstStride, 
                    (int*)srcOffsets.GetUnsafeReadOnlyPtr()
                );
                currentSplatOffset += splatsInThisChunk;
            }

            return dst;
        }

        [BurstCompile]
        static unsafe void ReorderPLYData(int splatCount, byte* src, int srcStride, byte* dst, int dstStride, int* srcOffsets)
        {
            for (int i = 0; i < splatCount; i++)
            {
                for (int attr = 0; attr < dstStride / 4; attr++)
                {
                    if (srcOffsets[attr] >= 0)
                        *(int*)(dst + attr * 4) = *(int*)(src + srcOffsets[attr]);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        [BurstCompile]
        struct LinearizeDataJob : IJobParallelFor
        {
            public NativeArray<InputSplatData> splatData;
            public void Execute(int index)
            {
                var splat = splatData[index];

                var q = splat.rot;
                var qq = GaussianUtils.NormalizeSwizzleRotation(new float4(q.x, q.y, q.z, q.w));
                splat.rot = new Quaternion(qq.x, qq.y, qq.z, qq.w);

                splat.scale = GaussianUtils.LinearScale(splat.scale);
                splat.dc0 = GaussianUtils.SH0ToColor(splat.dc0);
                splat.opacity = GaussianUtils.Sigmoid(splat.opacity);

                splatData[index] = splat;
            }
        }

        static void LinearizeData(NativeArray<InputSplatData> splatData)
        {
            LinearizeDataJob job = new LinearizeDataJob();
            job.splatData = splatData;
            job.Schedule(splatData.Length, 4096).Complete();
        }
    }
}
#endif
