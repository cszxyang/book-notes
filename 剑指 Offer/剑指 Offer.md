## 03. 数组中重复的数字

**题目描述：** 在一个长度为 n 的数组里的所有数字都在 0 到 n-1 的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组 {2, 3, 1, 0, 2, 5, 3}，那么对应的输出是第一个重复的数字2。

**知识点：** 数组

**分析** ：因为数组的数字在 0~n-1 范围，如果这个数组没有重复数字，那么这个数组排序后，数字 i 应该在下标为 i 的位置，由于数组有重复数字，所以有些位置可能存在多个数字，有些位置可能没有数字。

**方案** ：遍历数组，判断 num[i] == i 是否成立，是则继续扫描；否则，判断 num[i] == num[num[i]] 是否成立，是则说明存在重复的数字，否则交换 num[i] 与 num[num[i]]，相当于进行排序。

```java
public class Solution {
    // 时间复杂度：O(n)，空间复杂度：O(1)
    public boolean duplicate(int numbers[],int length,int [] duplication) {
        if (numbers == null || length <= 0)
            return false;
        // 排异
        for (int i = 0; i < length; i++) {
            if (numbers[i] < 0 || numbers[i] > length - 1)
                return false;
        }
        
        for (int i = 0, j = 0; i < length; i++) {
            while (numbers[i] != i) { // 每轮最多执行两次
                if (numbers[i] == numbers[numbers[i]]) {
                    duplication[j++] = numbers[i];
                    return true;
                }
                int temp = numbers[i];
                numbers[i] = numbers[temp];
                numbers[temp] = temp;
            }
        }
        return false;
    }    
}
```



## 04. 二维数组中的查找

**题目描述：** 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

**知识点：** 数组、查找

**分析：** 要在有序的规则数组中查找某个元素，很自然让人想到二分查找，而二分查找的思想说到底就是排除不可能的数字，缩小搜索范围，在一维数组中很自然地会选择中点进行判断，而在二维数组中选择入手点非常关键。

**方案：** 先选取数组右上角的数字，若该数字等于要查找的数字，则查找结束；如果该数字大于要查找的数字，说明该数字所在的列都大于要查找的数字，所以可以剔除该列；同理，如果该数字小于要查找的数字，说明该数字所在的行都小于要查找的数字，所以可以剔除该行。如此重复，直到穷尽。

```java
public class Solution {
    // 时间复杂度：O(n)，空间复杂度：O(1)
    public boolean Find(int target, int [][] array) {
        boolean found = false;
        if (array != null && array.length > 0) {
            int rows = array.length, cols = array[0].length;
            int row = 0, col = cols - 1;
            while (row < rows && col >= 0) {
                if (array[row][col] == target) {
                    found = true;
                    break;
                }
                else if (array[row][col] > target)  col--; // 排除列
                else row++; // 排除行
            }
        }
        return found;
    }
}
```

## 05. 替换空格

**题目描述：** 请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为 We Are Happy，则经过替换之后的字符串为 We%20Are%20Happy 。

**知识点：** 字符串



```java
public class Solution {
  public String replaceSpace(StringBuffer str) {
        char[] chars = str.toString().toCharArray();
        int blankCount = 0, originalLength = chars.length;
        for (char c : chars) {
            if (c == ' ')   blankCount++;
        }
        int newLength = originalLength + blankCount * 2;
        char[] newChars = new char[newLength];
        System.arraycopy(chars, 0, newChars, 0, originalLength);
        int indexOfOriginal = originalLength - 1, indexOfNew = newLength - 1;
        while (indexOfOriginal >= 0 && indexOfNew > indexOfOriginal) {
            if (newChars[indexOfOriginal] == ' ') {
                newChars[indexOfNew--] = '0';
                newChars[indexOfNew--] = '2';
                newChars[indexOfNew--] = '%';
            } else {
                newChars[indexOfNew--] = newChars[indexOfOriginal];
            }
            indexOfOriginal--;
        }
        return String.valueOf(newChars);
    }
}
```



## 06. 从尾到头打印单链表

**题目描述：** 输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。

**知识点：** 链表、递归

**分析：** 因为单链表结构的特殊性，使用迭代的方法是很难入手这个问题的，除非利用栈的先进后出的特点，而递归的本质就是隐式地使用了栈，所以可以考虑使用递归。

**方案：** 从头结点开始，每打印一个链表结点，先去打印这个结点的下一个结点，以结点为空作为递归终止的条件。

```java
import java.util.ArrayList;
public class Solution {
    // 牛客
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> result = new ArrayList<>();
        core(listNode, result);
        return result;
    }
    
    private void core(ListNode node, ArrayList<Integer> list) {
        if (node != null) {
            core(node.next, list);
            list.add(node.val);
        }
    }
    
    public void print(ListNode head) {
        if (head != null) {
            if (head.next != null) {
                print(head.next);
            }
            System.out.println(head.val);
        }
    }
}
```



## 07. 用前序和中序序列重建二叉树

**题目描述：** 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

**知识点：** 二叉树

**方案：** 前序序列的第一个数字就是根结点的值，扫描中序序列，就能确定根结点的位置，从而确定左右子树，如此递归即可。

```java
public class Solution {
    public TreeNode reConstructBinaryTree(int [] pre, int [] in) {
        if (pre == null || in == null || pre.length != in.length || 
            pre.length <= 0 || in.length <= 0) {
            return null;
        }
        return construct(pre, 0, pre.length - 1,
                  in, 0, in.length - 1);
    }
 
    private TreeNode construct (int[] preorder, int preStart, int preEnd,
                           int[] inorder, int inStart, int inEnd) {
        TreeNode root = new TreeNode(preorder[preStart]);
        if (preorder[preStart] == preorder[preEnd] &&
                inorder[inStart] == inorder[inEnd]) {
            return root;
        }
       
        int rootInorder = inStart;
        // 在中序序列中找到root的索引
        while (rootInorder < inEnd && inorder[rootInorder] != root.val)
            ++ rootInorder;
        
        // 在中序序列中根结点与有效起始索引间的距离，即root的左子树的结点个数
        int leftLength = rootInorder - inStart;
        // 计算在前序序列中左子树的结束点，加1就得到右子树在前序序列中的起始索引
        int leftPreorderEnd = preStart + leftLength;
        if (leftLength > 0) {   // 构建root的左子树
            root.left = construct(preorder, preStart + 1, leftPreorderEnd,
                    inorder, inStart, rootInorder - 1);
        }
        if (leftLength < preEnd - preStart) {   // 构建root的右子树
            root.right = construct(preorder, leftPreorderEnd + 1, preEnd,
                    inorder, rootInorder + 1, inEnd);
        }
        return root;
    }
}
```



## 08. 二叉树的下一个结点

**题目描述：** 给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

```java
public class TreeLinkNode {
    int val;
    TreeLinkNode left, right, parent;

    TreeLinkNode(int val) {
        this.val = val;
    }
}
```

**知识点：** 二叉树

**分析与方案：** 根据中序遍历的特点，

1. 判断该结点是否有右子树
   - 有右子树，那么它的下一个结点就是他的右子树的最左结点；
   - 没有右子树，判断它是其父结点的左结点还是右结点
     - 它是其父结点的左结点，则它的父结点是下一个结点
     - 它是其父结点的右结点，沿着其父结点向上找，直到找到一个是它父结点的左子结点的结点。

```java
public class Solution {
    public TreeLinkNode GetNext(TreeLinkNode pNode) {
        if (pNode == null)  return null;
        TreeLinkNode next = null;
        if (pNode.right != null) {
            TreeLinkNode right = pNode.right;
            while (right.left != null) {
                right = right.left;
            }
            next = right;
        }
        else if (pNode.next != null) {
            TreeLinkNode cur = pNode, parent = pNode.parent;
            while (parent != null && cur == parent.right) {
                cur = parent;
                parent = parent.parent;
            }
            next = parent;
        }
        return next;
    }
}
```



## 09. 用两个栈实现队列

**题目描述：** 用两个栈来实现一个队列，完成队列的 Push 和 Pop 操作。 队列中的元素为 int 类型。

**知识点：** 栈

**分析：** 栈的特点是先进后出，而队列的特点是先进先出，要想用两个栈实现队列，明显有点负负得正的意思。

**方案：** 关键在于 pop 出来的数据顺序符合队列的要求，可以使用 pushStack 记录入队列的数据顺序，在弹出操作开始时，先将 pushStack  中的数据灌进 popStack 中，经过两次入栈操作后的数据再出栈就符合队列的先进先出特点了。

```java
import java.util.Stack;

public class Solution {
    Stack<Integer> pushStack = new Stack<>();
    Stack<Integer> popStack = new Stack<>();

    public void push(int node) {
        pushStack.push(node);
    }

    public int pop() {
        if (popStack.isEmpty()) {
            while (!pushStack.isEmpty()) {
                popStack.push(pushStack.pop());
            }
        }
        return popStack.pop();
    }
}
```



## 10. 斐波那契数列

**题目描述：** 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0，n<=39）。

**知识点：** 递归

**分析：** 斐波那契数列的递推公式如下所示，很自然会想到直接根据公式分情况讨论即可，但第三条公式是从上往下递推的，它包含了很多重复的步骤。

<div align="center"> <img src="/assets/Fibonacci.jpg" width="500px"> </div><br>

**方案：** 使用从下往上计算，首先根据 f(0) 和 f(1) 算出 f(2)，再根据 f(1) 和 f(2) 算出 f(3)，以此类推就可以算出第 n 项了。

```java
public class Solution {
    public int Fibonacci(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;
        
        int fibMinusOne = 1; // f(n - 1)
        int fibMinusTwo = 0; // f(n - 2)
        int fibN = 0;
        for (int i = 2; i <= n; i++) {
            fibN = fibMinusOne + fibMinusTwo;
            fibMinusTwo = fibMinusOne;
            fibMinusOne = fibN;
        }
        return fibN;
    }
}
```

### 10.1 跳台阶

**题目描述：** 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

```java
public class Solution {
   public int JumpFloor(int target) {
        return Fibonacci(target);
    }

    private int Fibonacci(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;
        if (n == 2) return 2;

        int fibMinusOne = 2; // f(n - 1)
        int fibMinusTwo = 1; // f(n - 2)
        int fibN = 0;
        for (int i = 3; i <= n; i++) {
            fibN = fibMinusOne + fibMinusTwo;
            fibMinusTwo = fibMinusOne;
            fibMinusOne = fibN;
        }
        return fibN;
    }
}
```

### 10.2. 变态跳台阶

**题目描述：** 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

```java
public class Solution {
    public int JumpFloorII(int target) {
        return (int) Math.pow(2, target - 1);
    }
}
```

### 10.3. 矩阵覆盖

**题目描述：** 我们可以用 2\*1 的小矩形横着或者竖着去覆盖更大的矩形。请问用 n 个 2\*1 的小矩形无重叠地覆盖一个 2\*n 的大矩形，总共有多少种方法？

```java
public class Solution {
    public int RectCover(int target) {
        if (target == 0) return 0;
        if (target == 1) return 1;
        if (target == 2) return 2;
        int fibMinusOne = 2; // f(n - 1)
        int fibMinusTwo = 1; // f(n - 2)
        int fibN = 0;
        for (int i = 3; i <= target; i++) {
            fibN = fibMinusOne + fibMinusTwo;
            fibMinusTwo = fibMinusOne;
            fibMinusOne = fibN;
        }
        return fibN;
    }
}
```



## 11. 旋转数组中的最小数字

**题目描述：** 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

**知识点：** 数组、查找

**分析：** 所求的最小数字是旋转数组的分界；在排序数组中可使用二分法实现 O(lgn) 时间复杂度的查找。

**方案：** 

- 设置双指针分别从头和尾相向遍历
- 找到中间元素，分别判断中间元素 arr[mid] 与 arr[first]、arr[last] 的大小关系
  - 如果 arr[mid] >= arr[first]，说明中间元素位于前面的递增子序列，数组中的最小数字的索引 index => mid，所以置 first 为 mid，缩小范围继续查找。
  - 如果 arr[mid] <= arr[first]，说明中间元素位于后面的递增子序列，数组中的最小数字的索引 index <= mid，所以置 last 为 mid，缩小范围继续查找。
- 对于像 {1, 0, 1, 1, 1} 这样的特例，按照上面的步骤。会认为最小的数字在 arr[mid] 的后面，从而置 first 为 mid，但是这回错过真正最小的 0，这时只能进行顺序查找。

```java
public class Solution {
    public int minNumberInRotateArray(int [] array) {
        int first = 0, last = array.length - 1, mid = first;
        while (array[first] >= array[last]) {
            if (last - first == 1) {
                mid = last;
                break;
            }
            mid = (first + last) >> 1;
            if (array[mid] == array[last] && 
                    array[mid] == array[first]) {
                // 顺序查找
                return minInOrder(array, first, last);
            }
            if (array[mid] >= array[first])
                first = mid;
            else if (array[mid] <= array[last])
                last = mid;
        }
        return array[mid];
    }
    
    private int minInOrder (int[] arr, int first, int index2) {
        int result = arr[first];
        for (int i = first + 1; i <= index2; i++) {
            if (result > arr[i])
                result = arr[i];
        }
        return result;
    }
}
```



## 12. 矩阵中路径

**题目描述：** 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。**如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。** 例如 

```
a b c e 
s f c s 
a d e e
```

这样的 3 X 4 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。

**知识点：** 数组、回溯法

**方案：** 顺序遍历二维矩阵，对于每个位置 pos 判断其字符值是否为 str[pathLength]

- 是，递归判断 pos 四周是否能形成可达的完整路径，只要找到一条即返回
- 否，继续到下一个点进行判断

```java
public class Solution {
    int pathLength = 0;
    public boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
        if (matrix == null || rows <= 0 || cols <= 0 || str == null) {
            return false;
        }
        boolean[] visited = new boolean[rows * cols];
        
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (hasPathCore(matrix, rows, cols, row, col, str, visited))
                    return true;
            }
        }
        return false;
    }
    
    private boolean hasPathCore(char[] matrix, int rows, int cols, int row, int col, 
                                char[] str, boolean[] visited) {
        if (pathLength == str.length) {
            return true;
        }
        
        boolean hasPath = false;
        // 排除边缘点的某些不存在路径
        if (row >= 0 && row < rows && col >= 0 && col < cols && 
            matrix[row * cols + col] == str[pathLength] && !visited[row * cols + col]) {
            
            ++pathLength;
            visited[row * cols + col] = true;
            hasPath = hasPathCore(matrix, rows, cols, row, col-1, str, visited)
                || hasPathCore(matrix, rows, cols, row-1, col, str, visited)
                || hasPathCore(matrix, rows, cols, row, col+1, str, visited)
                || hasPathCore(matrix, rows, cols, row+1, col, str, visited);
            
            if (!hasPath) {
                --pathLength;
                visited[row * cols + col] = false;
            }
        }
        return hasPath;
    }
}
```



## 13. 机器人的运动范围

**题目描述：** 地上有一个 m 行和 n 列的方格。一个机器人从坐标 (0, 0) 的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是**不能进入行坐标和列坐标的数位之和大于 k 的格子**。 例如，当 k 为 18 时，机器人能够进入方格（35, 37），因为 3+5+3+7=18。但是，它不能进入方格（35, 38），因为 3+5+3+8=19。请问该机器人能够达到多少个格子？

**知识点：** 数组、回溯法

**分析：** 机器人从 (0, 0) 开始移动，当它准备进入 (i, j) 的格子时，先检查坐标的位数判断能否进入，若能，则继续判断 (i, j) 周围的点的位数。

```java
public class Solution {
    public int movingCount(int threshold, int rows, int cols) {
        if (threshold < 0 || rows <= 0 || cols <= 0)     return 0;
        return movingCountCore(threshold, rows, cols, 0, 0, new boolean[rows * cols]);
    }
    
    private int movingCountCore(int threshold, int rows, int cols,
                               int row, int col, boolean[] visited) {
        int count = 0;
        if (check(threshold, rows, cols, row, col, visited)) {
            visited[row *cols + col] = true;	// 移进一步，不成不用回退
            count = 1 + movingCountCore(threshold, rows, cols, row-1, col, visited) 
                    + movingCountCore(threshold, rows, cols, row, col-1, visited) 
                    + movingCountCore(threshold, rows, cols, row+1, col, visited)
                    + movingCountCore(threshold, rows, cols, row, col+1, visited);
        }
        return count;
    }
    
    // 判断 row 与 col 的位数是否小于阈值
    private boolean check(int threshold, int rows, int cols,
                               int row, int col, boolean[] visited) {
        if (row >= 0 && row < rows && col >= 0 && col < cols && getDigitSum(row) + 
           getDigitSum(col) <= threshold && !visited[row *cols + col]) {
            return true;
        }
        return false;
    }
    
    private int getDigitSum(int num) {
        int sum = 0;
        while (num > 0) {
            sum += num % 10;
            num /= 10;
        }
        return sum;
    }
}
```



## 14. 剪绳子



## 15. 二进制中 1 的个数

**题目描述：** 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

**知识点：** 进制转化、补码反码原码

**方案1：** 从右往左诸位比较该整数的比特并记录1的个数。

```java
public int NumberOf1(int n) {
    int count = 0;
    int flag = 1;
    while (flag != 0) {
        if ((n & flag) != 0) {
            count++;
        }
        flag <<= 1;
    }
    return count;
}
```

**方案2：** 把一个整数减去 1，再和原整数做与运算，会把该整数最右边的 1 变成 0，那个一个整数的二进制表示中有多少个 1，就可以进行多少次这样的操作。

```java
public int NumberOf1(int n) {
    int count = 0;
    while (n != 0) {
        ++count;
        n = (n - 1) & n;
    }
    return count;
}
```



## 16. 数值的整数次方

**题目描述：** 给定一个 double 类型的浮点数 base 和 int 类型的整数 exponent。求 base<sup>exponent</sup>。

**知识点：** 数学

**方案：** 结合公式

<div align="center"> <img src="/assets/p_16.jpg" width="500px"> </div><br>

```java
public class Solution {
    public double Power(double base, int exponent) {
        if (base == 0 && exponent < 0)  return 0;
        double resolt = powerCore(base, Math.abs(exponent));
        if (exponent < 0)   resolt = 1 / resolt;
        return resolt;
    }

    private double powerCore(double base, int exponent) {
        if (exponent == 0)  return 1;
        if (exponent == 1)  return base;
        double result = powerCore(base, exponent >> 1);
        result *= result;
        // 奇数次幂，还要再乘一个底数
        if ((exponent & 1) == 1)    result *= base;
        return result;
    }
}
```



## 17. 打印从1到最大的n位数

**题目描述：** 给定一个 double 类型的浮点数base和int类型的整数exponent。求base的exponent次方。

**知识点：** 数学



## 18. 删除链表的节点

### 18.1 删除排序链表中的重复节点

**题目描述：** 在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

```java
public class Solution {
    public ListNode deleteDuplication(ListNode pHead) {
        if (pHead == null || pHead.next == null)    return pHead;
        ListNode next = pHead.next;
        if (pHead.val == next.val) {
            // 跳过重复元素
            while (next != null && next.val == pHead.val) 
                next = next.next;
            return deleteDuplication(next);
        } else {
            pHead.next = deleteDuplication(pHead.next);
            return pHead;
        }
    }
}
```



## 19. 正则表达式匹配

**题目描述：** 请实现一个函数用来匹配包括'.'和'\*'的正则表达式。模式中的字符'.'表示任意一个字符，而 **'\*'表示它前面的字符可以出现任意次（包含0次）** 。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配

**知识点：** 字符串

**方案：** 逐个比较 str 和 pattern 的字符，

- 若模式的字符是 '.'，则它可以匹配字符串中的任意字符，
- 若模式的字符不是 '.'，而且 str 与 pattern 相应位置的字符相同，则它们相互匹配。

但模式的第二个字符是 ‘\*’ 时，在模式上向后移动两个字符，相当于 ‘\*’ 和它前面的字符被忽略了

```java
public class Solution {
    public boolean match(char[] str, char[] pattern) {
        if (str == null || pattern == null) {
            return false;
        }
        return matchCore(str, 0, pattern, 0);
    }
    
    private boolean matchCore(char[] str, int strIndex, char[] pattern, int patternIndex) {
        if (strIndex == str.length && patternIndex == pattern.length) {
            return true;
        }
        if (strIndex != str.length && patternIndex == pattern.length) {
            return false;
        }
        // 模式的下一个字符是*
        if ((patternIndex + 1) < pattern.length && pattern[patternIndex + 1] == '*') {
            // 匹配的字符相同，
            if ((strIndex != str.length && pattern[patternIndex] == str[strIndex]) 
                || (pattern[patternIndex] == '.' && strIndex != str.length)) {
                return matchCore(str, strIndex+1, pattern, patternIndex+2)
                    || matchCore(str, strIndex+1, pattern, patternIndex)
                    || matchCore(str, strIndex, pattern, patternIndex+2);
            }
            else {
                return matchCore(str, strIndex, pattern, patternIndex+2);
            }
        }
        
        if ((strIndex != str.length && pattern[patternIndex] == str[strIndex]) 
            || (pattern[patternIndex] == '.' && strIndex != str.length)) {
            return matchCore(str, strIndex+1, pattern, patternIndex+1);
        }
        return false;
    }
}
```



## 20. 表示数值的字符串

**题目描述：** 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。



## 21. 调整数组顺序是奇数位于偶数前面

**题目描述：** 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

**知识点：** 数组

**方案1：** 书上的思想是设置头尾双指针，前面的指针往后跳过奇数，直到遇到偶数时停下，后面的指针往前跳过偶数，直到遇到偶数时停下，然后两指针所指向的元素交换，两指针继续移动直至相遇。然而这样就无法保证奇数与奇数，偶数和偶数之间的相对位置不变了。

```java
public class Solution {
    public static void reOrderArray(int[] array) {
        int i = 0, j = array.length - 1;
        while(true) {
            while (i < j) {
                if (!isEven(array[i])) i++;
            }
            while (i < j) {
                if (isEven(array[j])) j--;
            }
            if (i >= j)  break;
            swap(array, i, j);
        }
    }
    
    private static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
    
    private static boolean isEven(int num) {
        return (num & 1) == 0;
    }
}
```

**方案2：** 计算原数组 orig 中奇数的个数，找到结果数组中的分界点，将原来的数组复制一份得到 copy，遍历 copy，遇到奇数/偶数则依次放在原数组奇数/偶数区域。

```java
public class Solution {
    public static void reOrderArray(int[] array) {
       // 奇数个数
        int oddCnt = 0;
        for (int val : array)
            if (val % 2 == 1)
                oddCnt++;
        int[] copy = array.clone();
        int i = 0, j = oddCnt;
        for (int num : copy) {
            if (num % 2 == 1)
                array[i++] = num;
            else
                array[j++] = num;
        }
    }
}
```



## 22. 链表中倒数第k个结点

**题目描述：** 输入一个链表，输出该链表中倒数第k个结点。

**知识点：** 链表

```java
public class Solution {
    public ListNode FindKthToTail(ListNode head, int k) {
        if (head == null || k == 0)   return null;

        ListNode fast = head, slow = head;
        for (int i = 0; i < k - 1; i++) {
            if (fast.next != null)
                fast = fast.next;
            else return null;
        }
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }
}
```



## 23. 链表中环的入口结点

**题目描述：** 给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。

**知识点：** 链表



## 24. 反转链表

**题目描述：** 输入一个链表，反转链表后，输出新链表的表头。

**知识点：** 链表

**方案1：** 增加两个指针记录遍历点的前继和后继结点，依次改变遍历点的前后指针的指向。

```java
public class Solution {
   public ListNode ReverseList(ListNode head) {
        if (head == null || head.next == null)  return head;
        ListNode next, prev = null;
        while (head != null) {
            next = head.next;
            head.next = prev;
            prev = head;
            head = next;
        }
        return prev;
    }
}
```

**方案2：** 增加一个伪头结点，利用栈的思想，依次将原链表中的每个结点插到伪头结点与其下一个结点间。

```java
public ListNode ReverseList(ListNode head) {
    if (head == null || head.next == null)  return head;
    ListNode next, newHead = new ListNode(-1);
    while (head != null) {
        next = head.next;
        head.next = newHead.next;
        newHead.next = head;
        head = next;
    }
    return newHead.next;
}
```

**方案3：** 通过递归从尾回头构造反转链表

```java
public ListNode ReverseList(ListNode head) {
    if (head == null || head.next == null)  return head;
    ListNode next = head.next;
    // 尾结点成为头结点，作为递归结果返回
    ListNode newHead = ReverseList(next);
    // 断旧立新
    next.next = head;
    head.next = null;
    return newHead;
}
```

**方案4：** 很无聊地，既然可以使用递归，那自然也能用栈实现，不过空间复杂度与时间复杂度可能不达标。

```java
public ListNode ReverseList(ListNode head) {
    if (head == null || head.next == null)  return head;
    Stack<ListNode> stack = new Stack<>();
    while (head != null) {
        stack.push(head);
        head = head.next;
    }
    ListNode newHead = new ListNode(-1), next = newHead;
    while (!stack.isEmpty()) {
        next.next = stack.pop();
        next = next.next;
    }
    return newHead.next;
}
```





## 25. 合并两个排序链表

**题目描述：** 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

**知识点：** 链表

**方案：** 当我们得到两个链表中较小的头结点并把它链接到已经合并的链表之后，两个链表剩余的结点依然是有序的，因此合并的步骤和之前的步骤是一样的，因此可以使用递归。

```
1->3->5->7					3->5->7						3->5->7
|							|							|
p1					1->		p1					1->2	p1
2->4->6->8					2->4->6->8					4->6->8
|							|							|
p2							p2							p2
```

```java
public class Solution {
   public ListNode Merge(ListNode list1, ListNode list2) {
        if (list1 == null)     return list2;
        else if (list2 == null)    return list1;

        ListNode merged = null;

        if (list1.val < list2.val) {
            merged = list1;
            merged.next = Merge(merged.next,list2);
        } else {
            merged = list2;
            merged.next = Merge(list1, merged.next);
        }
        return merged;
    }
}
```



## 26. 树的子结构

**题目描述：** 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

**知识点：** 树

**方案：** 

```java
public class Solution {
    public boolean HasSubtree(TreeNode root1,TreeNode root2) {
        boolean result = false;
        if (root1 != null && root2 != null) {
            if (root1.val == root2.val) 
                result = compareChildren(root1, root2);
            if (!result)
                result = HasSubtree(root1.left, root2);
            if (!result)
                result = HasSubtree(root1.right, root2);
        }
        return result;
    }

    private boolean compareChildren(TreeNode root1, TreeNode root2) {
        if (root2 == null)  return true;
        if (root1 == null)  return false;
        if (root1.val != root2.val)     return false;
        return compareChildren(root1.left, root2.left) && 
                compareChildren(root1.right, root2.right);
    }
}
```



## 27. 二叉树的镜像

**题目描述：** 操作给定的二叉树，将其变换为源二叉树的镜像。

```
二叉树的镜像定义：源二叉树   						镜像二叉树
    	    8										8
    	   /  \ 								   /  \
    	  6   10								  10   6
    	 / \  / \								 / \  / \
    	5  7 9 11								11 9 7  5
```

**知识点：** 树

**方案：** 先前序遍历这棵树的每个结点，如果遍历到的结点有子结点，就交换它的两个子结点，当交换完所有叶子结点的左右子结点后就得到树的镜像。

```java
public class Solution {
   public void Mirror(TreeNode root) {
        if (root != null) {
            if (root.left != null || root.right != null) {
                TreeNode temp = root.left;
                root.left = root.right;
                root.right = temp;
                
                if (root.left != null) {
                    Mirror(root.left);
                }
                if (root.right != null) {
                    Mirror(root.right);
                }
            }
        }
    }
}
```



## 28. 对称的二叉树

**题目描述：** 请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

**知识点：** 树

**方案：** 

```java
public class Solution {
   boolean isSymmetrical(TreeNode pRoot) {
        return isSymmetrical(pRoot, pRoot);
    }
    
    boolean isSymmetrical(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null)
            return true;
        if (root1 == null || root2 == null) 
            return false;
        if (root1.val != root2.val) 
            return false;
        return isSymmetrical(root1.left, root2.right) &&
                isSymmetrical(root1.right, root2.left);
    }
}
```



## 29. 顺时针打印矩阵

**题目描述：** 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

**知识点：** 

## 30. 包含 min 函数的栈

**题目描述：** 定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O(1)）。

**知识点：** 栈

**方案：** 

```java
import java.util.Stack;

public class Solution {
    
   private Stack<Integer> dataStack = new Stack<>();
    private Stack<Integer> auxStack = new Stack<>();

    public void push(int node) {
        dataStack.push(node);
        if (auxStack.size() == 0 || node < auxStack.peek())
            auxStack.push(node);
        else auxStack.push(auxStack.peek());
    }

    public void pop() {
        if (dataStack.size() > 0 && auxStack.size() > 0) {
            dataStack.pop();
            auxStack.pop();
        }
    }

    public int top() {
        return dataStack.peek();
    }

    public int min() {
        return auxStack.peek();
    }
}
```



## 31. 栈的压入、弹出序列

**题目描述：** 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

**知识点：** 栈

```java
import java.util.Stack;
public class Solution {
    public boolean IsPopOrder(int[] pushA, int[] popA) {
        boolean isPossible = false;
        if (pushA != null && popA != null) {
            int length = pushA.length;
            int nextPush = 0, nextPop = 0;
            Stack<Integer> auxStack = new Stack<>();
            while (nextPop < length) {
                while (auxStack.isEmpty() || auxStack.peek() != popA[nextPop]) {
                    if (nextPush == length) break;
                    auxStack.push(pushA[nextPush]);
                    nextPush++;
                }
                if (auxStack.peek() != popA[nextPop]) break;
                auxStack.pop();
                nextPop++;
            }
            if (auxStack.isEmpty() && nextPop == length)
                isPossible = true;
        }
        return isPossible;
    }
}
```



## 32. 从上到下打印二叉树

**题目描述：** 从上往下打印出二叉树的每个节点，同层节点从左至右打印。

**知识点：** 队列、树

```java
public class Solution {
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> res = new ArrayList<>();
        if(root == null)    return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()){
            TreeNode node = queue.remove();
            res.add(node.val);
            if(node.left != null)
                queue.add(node.left);
            if(node.right != null)
                queue.add(node.right);
        }
        return res;
    }
}
```

### 32.1 按之字形顺序打印二叉树

**题目描述：** 请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

**知识点：** 栈、树

```java
public class Solution {
    public ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(pRoot);
        boolean reverse = false;
        while (!queue.isEmpty()) {
            ArrayList<Integer> list = new ArrayList<>();
            int cnt = queue.size();
            while (cnt-- > 0) {
                TreeNode node = queue.poll();
                if (node == null)
                    continue;
                list.add(node.val);
                queue.add(node.left);
                queue.add(node.right);
            }
            if (reverse)
                Collections.reverse(list);
            reverse = !reverse;
            if (list.size() != 0)
                ret.add(list);
        }
        return ret;
    }
}
```





## 33. 二叉搜索树的后续遍历序列

**题目描述：** 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

**知识点：** 栈、树

```java
public class Solution {
    public boolean VerifySquenceOfBST(int[] sequence) {
        return VerifySequenceOfBST(sequence, 0, sequence.length);
    }

    private boolean VerifySequenceOfBST(int[] sequence, int start, int length) {
        if (sequence == null || length <= 0)
            return false;

        int rootVal = sequence[length - 1], i = start;
        for (; i < length - 1; i++) {
            if (sequence[i] > rootVal)  break;
        }
        int j = i;
        for (; j < length - 1; j++) {
            if (sequence[j] < rootVal)  return false;
        }

        boolean left = true, right = true;
        if (i > start)
            left = VerifySequenceOfBST(sequence, start, i);
        if (j < length - 1)
            right = VerifySequenceOfBST(sequence, i, length - 1 - j);
        return left && right;
    }
}
```



## 34. 二叉树中和为某一值的路径

**题目描述：** 输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)



## 35. 复杂链表的复制

输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）



## 36. 二叉搜索树与双向链表

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。



## 37. 序列化二叉树

请实现两个函数，分别用来序列化和反序列化二叉树



## 48. 最长不含重复字符的子串



## 50. 第一个只出现一次的字符

**题目描述：** 在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）。

**知识点：** 字符串

**分析：** 书中的思路是所有字符也不过 256 种，直接通过统计的方法记录每个字符出现的次数，返回统计数组中第一次出现的字符下标，然而作者却忽略了字符出现的顺序，即顺序遍历统计数组会得到一个出现一次的字符，但这个字符在原字符串中却不一定是最先出现的。

```java
public class Solution {
    public int FirstNotRepeatingChar(String str) {
        if (str == null)    return -1;
        int[] hashTable = new int[256];
        char[] chars = str.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            hashTable[chars[i]]++;
        }
        for (int i = 0; i < chars.length; i++) {
            if (hashTable[chars[i]] == 1) {
                return i;
            }
        }
        return -1;
    }
}
```

**方案：** 使用 LinkedHashMap 记录每个字符出现的次数及顺序。

```java
import java.util.LinkedHashMap;
public class Solution {
    public int FirstNotRepeatingChar(String str) {
        LinkedHashMap <Character, Integer> map = new LinkedHashMap<Character, Integer>();
        for(int i = 0;i < str.length(); i++){
            if(map.containsKey(str.charAt(i))){
                int times = map.get(str.charAt(i));
                map.put(str.charAt(i), ++times);
            }
            else {
                map.put(str.charAt(i), 1);
            }
        }
        int pos = -1;
        for(int i = 0; i < str.length(); i++){
            char c = str.charAt(i);
            if (map.get(c) == 1) {
                return i;
            }
        }
        return pos;
    }
}
```



