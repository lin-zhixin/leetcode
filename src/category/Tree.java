package category;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {
    }

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
};

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

public class Tree {
    public TreeNode buildBylevel(TreeNode root, int[] nums) {
        if (nums.length == 0) {
            return null;
        }
        root = new TreeNode(nums[0]);
        TreeNode r = root;
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(r);
        int i = 1;
        while (!queue.isEmpty()) {
            TreeNode tmp = queue.poll();
            if (i < nums.length && nums[i] != -1) {
                tmp.left = new TreeNode(nums[i]);
                queue.offer(tmp.left);
            }
            i++;
            if (i < nums.length && nums[i] != -1) {
                tmp.right = new TreeNode(nums[i]);
                queue.offer(tmp.right);
            }
            i++;
        }
        return root;
    }


    //     94. 二叉树的中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        inorderTraversalDG(root, res);
        return res;
    }

    //    非递归
    public List<Integer> inorderTraversal2(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        while (Objects.nonNull(root) || !stack.isEmpty()) {
            while (Objects.nonNull(root)) {
                stack.push(root);
                root = root.left;
            }
            res.add(stack.pop().val);
            root = root.right;
        }
        return res;
    }

    ///后序非递归
    public List<Integer> postOrder(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        TreeNode pre = null, now;
        while (!stack.isEmpty()) {
            now = stack.peek();
            if (Objects.isNull(now.left) && Objects.isNull(now.right) || (Objects.nonNull(pre) && (Objects.equals(now.left, pre) || Objects.equals(now.right, pre)))) {
                res.add(now.val);
                pre = stack.pop();
            } else {
                if (Objects.nonNull(now.right)) stack.push(now.right);
                if (Objects.nonNull(now.left)) stack.push(now.left);
            }
        }
        return res;
    }

    //后序非递归（栈存放祖先节点）
    public List<Integer> postOrder2(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode pre = null, now;
        while (Objects.nonNull(root) || !stack.isEmpty()) {
            if (Objects.nonNull(root)) {
                stack.push(root);
                root = root.left;
            } else {
                now = stack.peek();
                //Objects.nonNull(now.right)在访问左节点的时候会用到，now.right右边为空说明now没有哦右边也就是now该出来了  不能进入这个if  因为左右根的顺序 右边没有
                // !Objects.equals(now.right, pre)第二个条件是在now这个点二次被访问的时候，也就是now的右节点刚被访问完 此时now也该出来了 不能进入这个if 因为左右根的顺序
                if (Objects.nonNull(now.right) && !Objects.equals(now.right, pre)) {
                    root = now.right;
                } else {
                    res.add(stack.pop().val);
                    pre = now;
//                    root = null;
                }

            }
        }
        return res;
    }

    //101. 对称二叉树
    public boolean isSymmetric(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode u = queue.poll();
            TreeNode v = queue.poll();
            if (Objects.isNull(u) && Objects.isNull(v)) {
                continue;
            }
            if ((Objects.isNull(u) || Objects.isNull(v)) || !Objects.equals(u.val, v.val)) {
                return false;
            }
            queue.offer(u.left);
            queue.offer(v.right);
            queue.offer(u.right);
            queue.offer(v.left);
        }
        return true;
    }

    public void inorderTraversalDG(TreeNode root, List<Integer> res) {
        if (Objects.nonNull(root)) {
            inorderTraversalDG(root.left, res);
            res.add(root.val);
            inorderTraversalDG(root.right, res);
        }
    }

    //104. 二叉树的最大深度
    public int maxDepth(TreeNode root) {

        int max = 0;

        if (Objects.isNull(root)) {
            return 0;
        }
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int sz = queue.size();
            while (sz > 0) {
                root = queue.poll();
                if (Objects.nonNull(root.left)) {
                    queue.offer(root.left);
                }
                if (Objects.nonNull(root.right)) {
                    queue.offer(root.right);
                }
                sz--;
            }
            max++;
        }
        return max;
    }

    //108. 将有序数组转换为二叉搜索树
    public TreeNode sortedArrayToBST(int[] nums) {
        return helper(nums, 0, nums.length - 1);

    }

    public TreeNode helper(int[] nums, int left, int right) {
        if (left > right) {
            return null;
        }
        int mid = (left + right) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = helper(nums, left, mid - 1);
        root.right = helper(nums, mid + 1, right);
        return root;
    }

    //98. 验证二叉搜索树
    public boolean isValidBST(TreeNode root) {

        if (Objects.isNull(root)) {
            return true;
        }
        Stack<TreeNode> stack = new Stack<TreeNode>();

//        stack.push(root);
        long pre = -2147483649L;
        while (Objects.nonNull(root) || !stack.isEmpty()) {
            while (Objects.nonNull(root)) {
////                if (Objects.isNull(root.left) && Objects.isNull(root.right)) {
////                    continue;
////                }
//                if (Objects.isNull(root.left) && Objects.nonNull(root.right) && root.right.val <= root.val) {
//                    return false;
//                }
//                if (Objects.isNull(root.right) && Objects.nonNull(root.left) && root.left.val >= root.val) {
//                    return false;
//                }
//                if (Objects.nonNull(root.left) && Objects.nonNull(root.right) && (root.right.val <= root.val || root.left.val >= root.val || root.val)) {
//                    return false;
//                }
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (root.val <= pre) {
                return false;
            }
            pre = root.val;
            ;
            root = root.right;
        }
        return true;
    }

    //102. 二叉树的层序遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (Objects.isNull(root)) return res;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            List<Integer> tempres = new ArrayList<>();
            int sz = q.size();
            while (sz > 0) {
                TreeNode tmp = q.poll();
                tempres.add(tmp.val);
                if (Objects.nonNull(tmp.left)) q.offer(tmp.left);
                if (Objects.nonNull(tmp.right)) q.offer(tmp.right);
                sz--;
            }
            res.add(tempres);

        }
        return res;

    }

    //230. 二叉搜索树中第K小的元素
    public int kthSmallest(TreeNode root, int k) {

        if (Objects.isNull(root)) {
            return 0;
        }
        Stack<TreeNode> stack = new Stack<>();
//        stack.push(root);
        int count = 1;
        while (Objects.nonNull(root) || !stack.isEmpty()) {
            while (Objects.nonNull(root)) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (k == count++) {
                return root.val;
            }
            root = root.right;
        }
        return 0;
    }

    public Node connect(Node root) {
        List<List<Integer>> res = new ArrayList<>();
        if (Objects.isNull(root)) return root;
        Queue<Node> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            List<Integer> tempres = new ArrayList<>();
            int sz = q.size();
            Node pre = q.peek();
            while (sz > 0) {
                Node now = q.poll();
//                tempres.add(now.val);
                if (Objects.nonNull(now.left)) q.offer(now.left);
                if (Objects.nonNull(now.right)) q.offer(now.right);
                pre.next = now;
                pre = now;
                sz--;
                if (sz == 0) now.next = null;
            }
            res.add(tempres);
        }
        return root;

    }

    //236. 二叉树的最近公共祖先
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
//        Comparator
        Stack<TreeNode> stack = new Stack<TreeNode>();
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        TreeNode r = root, top, pre = null;
//        stack.push(r);
        while (Objects.nonNull(r) || !stack.isEmpty()) {
            if (Objects.nonNull(r)) {
                stack.push(r);
                r = r.left;
            } else {
                top = stack.peek();
                if (Objects.equals(top.val, p.val)) {
                    map = stack.stream().map(e -> e.val).collect(Collectors.toMap(k -> k, Function.identity()));
                }
                if (Objects.nonNull(top.right) && !Objects.equals(top.right, pre)) {
                    r = top.right;
                } else {
                    pre = stack.pop();
                }
            }
        }
        r = root;
        pre = null;
        while (Objects.nonNull(r) || !stack.isEmpty()) {
            if (Objects.nonNull(r)) {
                stack.push(r);
                r = r.left;
            } else {
                top = stack.peek();
                if (Objects.equals(top.val, q.val)) {
                    while (!stack.isEmpty()) {
                        top = stack.pop();
                        if (map.containsKey(top.val)) {
                            return top;
                        }
                    }
                }
                if (Objects.nonNull(top.right) && !Objects.equals(top.right, pre)) {
                    r = top.right;
                } else {
                    pre = stack.pop();
                }
            }
        }

        return root;

    }


    //226. 翻转二叉树
    public TreeNode invertTree(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root, top, pre = null;
        while (Objects.nonNull(p) || !stack.isEmpty()) {
            if (Objects.nonNull(p)) {
                stack.push(p);
                p = p.left;
            } else {
                top = stack.peek();
                if (Objects.nonNull(top.right) && !Objects.equals(pre, top.right)) {
                    p = top.right;
                } else {
                    stack.pop();
                    pre = top;
                    TreeNode t = top.left;
                    top.left = top.right;
                    top.right = t;
                }
            }
        }
        return root;

    }

    public TreeNode invertTree2(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root, top, pre = null;
        while (Objects.nonNull(p) || !stack.isEmpty()) {
            if (Objects.nonNull(p)) {
                stack.push(p);
                p = p.left;
            } else {
                top = stack.pop();
                p = top.right;
                TreeNode t = top.left;
                top.left = top.right;
                top.right = t;

            }
        }
        return root;

    }
//    剑指 Offer 55 - I. 二叉树的深度

    public int maxDepth2(TreeNode root) {
        Stack<TreeNode> stack = new Stack<TreeNode>();
        TreeNode r = root, pre = null, top;
        int max = 0;
        while (Objects.nonNull(r) || !stack.isEmpty()) {
            if (Objects.nonNull(r)) {
                stack.push(r);
                max = Math.max(max, stack.size());
                r = r.left;
            } else {
                top = stack.peek();
                if (Objects.nonNull(top.right) && !Objects.equals(top.right, pre)) {
                    r = top.right;
                } else {
                    pre = stack.pop();
                }
            }
        }
        return max;

    }

    int max = 0;

    public int diameterOfBinaryTree(TreeNode root) {
        preOrder(root);
        return max - 1;
    }

    public int preOrder(TreeNode root) {
        if (Objects.equals(null, root)) {
            return 0;
        }
        int l = preOrder(root.left);
        int r = preOrder(root.right);
        max = Math.max(max, l + r + 1);
//        System.out.println(max);
        return Math.max(l, r) + 1;
    }

    //617. 合并二叉树
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        TreeNode root3 = mergeTrees(root1, root2, null, 0);
//        System.out.println(inorderTraversal(mergeTrees(root1, root2, null,0)));
        return root3;
    }

    public TreeNode mergeTrees(TreeNode root1, TreeNode root2, TreeNode root3, int n) {
        if (Objects.nonNull(root1) || Objects.nonNull(root2)) {
//            if (Objects.nonNull(root1))
            Integer val = 0;
            if (Objects.nonNull(root2)) {
                val += root2.val;
            }
            if (Objects.nonNull(root1)) {
                val += root1.val;
            }
//            MyUtile.printBlank(n++,val.toString());

            root3 = new TreeNode(val);
            root3.left = mergeTrees(Optional.ofNullable(root1).map(r -> r.left).orElse(null), Optional.ofNullable(root2).map(r -> r.left).orElse(null), null, n);
            root3.right = mergeTrees(Optional.ofNullable(root1).map(r -> r.right).orElse(null), Optional.ofNullable(root2).map(r -> r.right).orElse(null), null, n);
//            MyUtile.printBlank(--n,val.toString());
            return root3;
        } else {
            return null;
        }
    }

    public static void main(String[] args) {
        Tree o = new Tree();
        TreeNode root1 = o.buildBylevel(new TreeNode(), new int[]{1, 3, 2, 5});
        TreeNode root2 = o.buildBylevel(new TreeNode(), new int[]{2, 1, 3, -1, 4, -1, 7});

//        System.out.println(o.inorderTraversal(root));
//        System.out.println(o.maxDepth(root));
//        System.out.println(o.isValidBST(root));
//        System.out.println(o.levelOrder(root));
//        System.out.println(o.lowestCommonAncestor(root, new TreeNode(5), new TreeNode(1)).val);
//        System.out.println(o.postOrder(root));
//        System.out.println(o.diameterOfBinaryTree(root));
        o.mergeTrees(root1, root2);
    }

}
